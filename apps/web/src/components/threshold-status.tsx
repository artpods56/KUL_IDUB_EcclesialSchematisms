import type { ReactNode } from "react";

import { BrandLoader, BrandWordmark } from "@/components/brand";

type ThresholdStatusProps = {
  title: string;
  detail: string;
  action?: ReactNode;
  loading?: boolean;
};

export function ThresholdStatus({
  title,
  detail,
  action,
  loading = false,
}: ThresholdStatusProps) {
  return (
    <main className="ns-auth-threshold">
      <div className="ns-auth-threshold__panel">
        <div className="ns-auth-threshold__brand">
          {loading ? (
            <BrandLoader size={88} label={title} />
          ) : (
            <BrandWordmark height={72} />
          )}
        </div>
        <div className="ns-auth-threshold__rule" aria-hidden="true" />
        <div className="ns-auth-threshold__copy">
          <h1>{title}</h1>
          <p className="ns-auth-threshold__detail">{detail}</p>
          {action ? <div className="ns-auth-threshold__action">{action}</div> : null}
        </div>
      </div>
    </main>
  );
}
