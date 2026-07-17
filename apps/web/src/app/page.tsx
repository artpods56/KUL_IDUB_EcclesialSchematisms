import { redirect } from "next/navigation";

import {
  LOCAL_WORKSPACE_SLUG,
  NEW_GRAPH_ROUTE_ID,
  workbenchGraphPath,
} from "@/components/workbench/routes";

export default function Home() {
  redirect(workbenchGraphPath(LOCAL_WORKSPACE_SLUG, NEW_GRAPH_ROUTE_ID));
}
