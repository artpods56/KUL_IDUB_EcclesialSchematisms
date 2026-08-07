export type ArtifactInteractionScalar = string | number | boolean | null;

export type ArtifactViewerEffect = "filter" | "highlight" | "focus";

export interface ArtifactViewerFieldMapping {
  sourceField: string;
  targetField: string;
}

export interface ArtifactViewerBinding {
  id: string;
  sourceViewerId: string;
  targetViewerId: string;
  mappings: ArtifactViewerFieldMapping[];
  effects: ArtifactViewerEffect[];
  emptySelection: "show_all";
}

export interface ArtifactKeySelectionItem {
  values: Record<string, ArtifactInteractionScalar>;
  sourceIndex?: number;
}

export interface ArtifactKeySelection {
  kind: "key-selection";
  items: ArtifactKeySelectionItem[];
}

export interface ArtifactViewerIncomingBinding {
  bindingId: string;
  effects: ArtifactViewerEffect[];
  sourceSelectionCount: number;
  rows: Array<Record<string, ArtifactInteractionScalar>>;
}

export interface ArtifactViewerActivity {
  state: "working" | "success" | "warning" | "error";
  title: string;
  message: string;
  retry?: () => void;
}

export interface ArtifactInteractionField {
  id: string;
  title: string;
  valueType: string;
}

export interface ArtifactViewerInteractionContext {
  outgoingFields: string[];
  selection: ArtifactKeySelection;
  incoming: ArtifactViewerIncomingBinding[];
  onFieldsChange: (fields: ArtifactInteractionField[]) => void;
  onSelectionChange: (selection: ArtifactKeySelection) => void;
  onActivityChange: (activity: ArtifactViewerActivity | null) => void;
}

export const EMPTY_ARTIFACT_KEY_SELECTION: ArtifactKeySelection = {
  kind: "key-selection",
  items: [],
};

export function targetRowsForBinding(
  binding: ArtifactViewerBinding,
  selection: ArtifactKeySelection,
): Array<Record<string, ArtifactInteractionScalar>> {
  if (
    binding.mappings.length === 0 ||
    binding.mappings.some(
      (mapping) => !mapping.sourceField || !mapping.targetField,
    )
  ) {
    return [];
  }

  return selection.items.flatMap((item) => {
    const row: Record<string, ArtifactInteractionScalar> = {};
    for (const mapping of binding.mappings) {
      if (!(mapping.sourceField in item.values)) return [];
      row[mapping.targetField] = item.values[mapping.sourceField] ?? null;
    }
    return [row];
  });
}
