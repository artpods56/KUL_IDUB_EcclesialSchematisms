import asyncio
from hashlib import sha256
from ipaddress import ip_address
import socket
import ssl

import httpx
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    TypeAdapter,
    field_validator,
)
import truststore

from notarius_core.artifacts import JsonObject

from notarius_plugin_gis.models import (
    Bounds,
    GeoFeatureCollection,
    validated_public_service_url,
)


_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


async def _resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    address_info = await asyncio.get_running_loop().getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
    )
    resolved_addresses = tuple(
        sorted({address[4][0].split("%", maxsplit=1)[0] for address in address_info})
    )
    if not resolved_addresses:
        raise ValueError(f"host {host!r} did not resolve to any address")
    for resolved_address in resolved_addresses:
        if not ip_address(resolved_address).is_global:
            raise ValueError(
                f"host {host!r} resolved to non-public address {resolved_address!r}"
            )
    return resolved_addresses


class WfsImportError(RuntimeError):
    pass


class _WfsFeaturePage(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    features: list[JsonObject]
    number_matched: StrictInt | None = Field(default=None, alias="numberMatched")
    number_returned: StrictInt | None = Field(default=None, alias="numberReturned")

    @field_validator("number_matched", "number_returned", mode="before")
    @classmethod
    def parse_counts(cls, value: JsonValue | None) -> JsonValue | None:
        if value in {None, "unknown"}:
            return None
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return value


class WfsClient:
    """Fetches bounded WFS 2.0 GeoJSON pages from a real remote service."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def fetch_feature_collection(
        self,
        *,
        service_url: str,
        type_name: str,
        source_name: str,
        page_size: int,
        max_page_bytes: int,
        timeout_seconds: float,
        max_features: int | None,
        bbox: Bounds | None,
        sort_by: str | None = None,
    ) -> GeoFeatureCollection:
        try:
            parsed_url = _HTTP_URL_ADAPTER.validate_python(service_url)
            validated_public_service_url(parsed_url, service_name="WFS")
        except ValueError as exc:
            raise WfsImportError(
                f"Rejected WFS 2.0 service endpoint {service_url!r} for type "
                f"{type_name!r}: {exc}"
            ) from exc
        service_url = str(parsed_url)
        service_host = parsed_url.host
        service_port = parsed_url.port
        if service_host is None or service_port is None:
            raise WfsImportError(
                f"Rejected WFS 2.0 service endpoint {service_url!r} for type "
                f"{type_name!r}: URL must include a host and port"
            )

        if page_size < 1:
            raise ValueError("WFS page_size must be positive")
        if max_features is not None and max_features < 1:
            raise ValueError("WFS max_features must be positive when provided")
        if max_page_bytes < 1:
            raise ValueError("WFS max_page_bytes must be positive")
        if max_features is not None and page_size > max_features:
            page_size = max_features

        features: list[JsonObject] = []
        start_index = 0
        previous_page_hash: str | None = None
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        ) as client:
            while max_features is None or len(features) < max_features:
                request_count = page_size
                if max_features is not None:
                    request_count = min(
                        page_size,
                        max_features - len(features),
                    )
                params: dict[str, str | int] = {
                    "service": "WFS",
                    "version": "2.0.0",
                    "request": "GetFeature",
                    "typeNames": type_name,
                    "outputFormat": "application/json",
                    "srsName": "EPSG:4326",
                    "count": request_count,
                    "startIndex": start_index,
                }
                if sort_by is not None:
                    params["sortBy"] = sort_by
                if bbox is not None:
                    params["bbox"] = ",".join(str(coordinate) for coordinate in bbox)
                    params["bbox"] += ",EPSG:4326"

                try:
                    await _resolve_public_addresses(service_host, service_port)
                except (OSError, ValueError) as exc:
                    raise WfsImportError(
                        f"Rejected WFS 2.0 GetFeature from {service_url!r} for "
                        f"type {type_name!r} at startIndex={start_index}, "
                        f"count={request_count} after runtime DNS validation: {exc}"
                    ) from exc

                try:
                    async with client.stream(
                        "GET",
                        service_url,
                        params=params,
                        headers={"Accept": "application/geo+json, application/json"},
                    ) as response:
                        response.raise_for_status()
                        response_bytes = bytearray()
                        async for chunk in response.aiter_bytes():
                            response_bytes.extend(chunk)
                            if len(response_bytes) > max_page_bytes:
                                raise WfsImportError(
                                    f"WFS 2.0 GetFeature from {service_url!r} for "
                                    f"type {type_name!r} at startIndex={start_index} "
                                    f"exceeded the {max_page_bytes}-byte page limit"
                                )
                except WfsImportError:
                    raise
                except httpx.HTTPStatusError as exc:
                    raise WfsImportError(
                        f"Failed WFS 2.0 GetFeature from {service_url!r} for "
                        f"type {type_name!r} at startIndex={start_index}, "
                        f"count={request_count}, status={exc.response.status_code}"
                    ) from exc
                except httpx.HTTPError as exc:
                    raise WfsImportError(
                        f"Failed WFS 2.0 GetFeature from {service_url!r} for "
                        f"type {type_name!r} at startIndex={start_index}, "
                        f"count={request_count}: {exc}"
                    ) from exc

                try:
                    page = _WfsFeaturePage.model_validate_json(response_bytes)
                except Exception as exc:
                    raise WfsImportError(
                        f"WFS 2.0 GetFeature from {service_url!r} for type "
                        f"{type_name!r} at startIndex={start_index} returned "
                        "invalid GeoJSON"
                    ) from exc

                if page.type != "FeatureCollection":
                    raise WfsImportError(
                        f"WFS 2.0 GetFeature from {service_url!r} for type "
                        f"{type_name!r} returned GeoJSON type {page.type!r}, "
                        "expected 'FeatureCollection'"
                    )
                if page.number_returned is not None and page.number_returned != len(
                    page.features
                ):
                    raise WfsImportError(
                        f"WFS 2.0 GetFeature from {service_url!r} for type "
                        f"{type_name!r} reported numberReturned={page.number_returned} "
                        f"but included {len(page.features)} features"
                    )
                if len(page.features) > request_count:
                    raise WfsImportError(
                        f"WFS 2.0 GetFeature from {service_url!r} for type "
                        f"{type_name!r} returned {len(page.features)} features, "
                        f"exceeding requested count={request_count}"
                    )
                if not page.features:
                    break

                page_hash = sha256(response_bytes).hexdigest()
                if previous_page_hash == page_hash:
                    raise WfsImportError(
                        f"WFS 2.0 service {service_url!r} ignored paging for type "
                        f"{type_name!r}; repeated page at startIndex={start_index}"
                    )
                previous_page_hash = page_hash
                features.extend(page.features)
                start_index += len(page.features)

                if (
                    page.number_matched is not None
                    and start_index >= page.number_matched
                ):
                    break
                if len(page.features) < request_count:
                    break

        try:
            return GeoFeatureCollection.from_features(features, source_name)
        except ValueError as exc:
            raise WfsImportError(
                f"WFS 2.0 GetFeature from {service_url!r} for type {type_name!r} "
                "did not return a valid EPSG:4326 GeoJSON FeatureCollection"
            ) from exc


__all__ = ["WfsClient", "WfsImportError"]
