"use client";

import * as React from "react";

/**
 * Collaborator selection ring that stays even under React Flow zoom.
 *
 * CSS box-shadow / outline rings paint in transformed space and routinely
 * drop a full edge at fractional scales (left gone, right still visible).
 * SVG `vector-effect: non-scaling-stroke` keeps a constant screen-pixel stroke.
 */
export function RemoteSelectionRing({
  color,
  radius = 16,
}: {
  color: string;
  radius?: number;
}) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [size, setSize] = React.useState({ w: 0, h: 0 });

  React.useLayoutEffect(() => {
    const parent = svgRef.current?.parentElement;
    if (!parent) return;

    const publish = () => {
      setSize({ w: parent.clientWidth, h: parent.clientHeight });
    };
    publish();

    const observer = new ResizeObserver(publish);
    observer.observe(parent);
    return () => observer.disconnect();
  }, []);

  const inset = 1;
  const w = size.w;
  const h = size.h;
  const show = w > inset * 2 && h > inset * 2;

  return (
    <svg
      ref={svgRef}
      aria-hidden
      width={show ? w : 0}
      height={show ? h : 0}
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        overflow: "visible",
        pointerEvents: "none",
        zIndex: 6,
      }}
    >
      {show ? (
        <>
          {/* Soft bloom — also non-scaling so zoom-out doesn't erase one side. */}
          <rect
            x={inset}
            y={inset}
            width={w - inset * 2}
            height={h - inset * 2}
            rx={Math.max(0, radius - inset)}
            ry={Math.max(0, radius - inset)}
            fill="none"
            stroke={color}
            strokeOpacity={0.22}
            strokeWidth={5}
            style={{ vectorEffect: "non-scaling-stroke" }}
          />
          <rect
            x={inset}
            y={inset}
            width={w - inset * 2}
            height={h - inset * 2}
            rx={Math.max(0, radius - inset)}
            ry={Math.max(0, radius - inset)}
            fill="none"
            stroke={color}
            strokeOpacity={0.55}
            strokeWidth={2}
            style={{ vectorEffect: "non-scaling-stroke" }}
          />
        </>
      ) : null}
    </svg>
  );
}
