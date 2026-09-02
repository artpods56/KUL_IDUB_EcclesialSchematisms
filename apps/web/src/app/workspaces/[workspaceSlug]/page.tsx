import { redirect } from "next/navigation";

interface WorkspacePageProps {
  params: Promise<{ workspaceSlug: string }>;
}

export default async function WorkspacePage({ params }: WorkspacePageProps) {
  const { workspaceSlug } = await params;
  redirect(`/workspaces/${encodeURIComponent(workspaceSlug)}/settings`);
}
