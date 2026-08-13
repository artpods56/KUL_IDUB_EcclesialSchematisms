import { notFound } from "next/navigation";

import { getSpike } from "@/sandbox/catalog";
import { SpikeHost } from "@/sandbox/SpikeHost";

interface SpikePageProps {
  params: Promise<{ spike: string }>;
}

export default async function SpikePage({ params }: SpikePageProps) {
  const { spike } = await params;
  const entry = getSpike(spike);
  if (!entry) {
    notFound();
  }
  return <SpikeHost spikeId={entry.id} />;
}
