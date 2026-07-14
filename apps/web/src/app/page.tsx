import { Workbench } from "@/components/workbench/Workbench";
import { LOCAL_WORKSPACE_SLUG } from "@/components/workbench/routes";

export default function Home() {
  return (
    <Workbench
      workspaceSlug={LOCAL_WORKSPACE_SLUG}
      initialGraphId={null}
      seedExample
    />
  );
}
