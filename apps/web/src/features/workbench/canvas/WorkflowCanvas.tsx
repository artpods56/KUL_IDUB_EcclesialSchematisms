"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  type EdgeTypes,
  type FitViewOptions,
  type IsValidConnection,
  type NodeTypes,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
  type ReactFlowInstance,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useTheme } from "@/components/theme";
import { tokens } from "@/lib/stylex/tokens.stylex";
import {
  ARTIFACT_VIEWER_EDGE_TYPE,
  ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE,
  ARTIFACT_VIEWER_NODE_TYPE,
  type CanvasEdge,
  type CanvasNode,
} from "./artifact-viewer";
import { connectionIsValid } from "./handles";
import ArtifactViewerEdge from "./edges/ArtifactViewerEdge";
import ArtifactViewerInteractionEdge from "./edges/ArtifactViewerInteractionEdge";
import WorkflowEdgeControl from "./edges/WorkflowEdge";
import ArtifactViewerNode from "./nodes/ArtifactViewerNode";
import WorkflowNodeCard from "./nodes/WorkflowNode";
import {
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  type WorkflowEdge,
} from "./types";

export const nodeTypes: NodeTypes = {
  [WORKFLOW_NODE_TYPE]: WorkflowNodeCard,
  [ARTIFACT_VIEWER_NODE_TYPE]: ArtifactViewerNode,
};

export const edgeTypes: EdgeTypes = {
  [WORKFLOW_EDGE_TYPE]: WorkflowEdgeControl,
  [ARTIFACT_VIEWER_EDGE_TYPE]: ArtifactViewerEdge,
  [ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE]: ArtifactViewerInteractionEdge,
};

const s = stylex.create({
  wrapper: {
    position: "relative",
    width: "100%",
    height: "100%",
    backgroundColor: tokens.colorBg,
  },
});

export interface WorkflowCanvasProps {
  children?: React.ReactNode;
  fitViewOptions?: FitViewOptions<CanvasNode>;
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  onNodesChange: OnNodesChange<CanvasNode>;
  onEdgesChange: OnEdgesChange<CanvasEdge>;
  onConnect: OnConnect;
  isValidConnection?: IsValidConnection<CanvasEdge>;
  onPaneReady?: (
    instance: ReactFlowInstance<CanvasNode, CanvasEdge>,
  ) => void;
  onPaneClick?: () => void;
  animateEdges?: boolean;
}

export function WorkflowCanvas({
  children,
  fitViewOptions,
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  isValidConnection = connectionIsValid,
  onPaneReady,
  onPaneClick,
  animateEdges = false,
}: WorkflowCanvasProps) {
  const { resolved } = useTheme();
  const renderedEdges = React.useMemo(
    () => edges.map((edge) => ({
      ...edge,
      animated:
        edge.type === WORKFLOW_EDGE_TYPE &&
        animateEdges &&
        !(edge as WorkflowEdge).data?.compatibilityIssues?.length,
    })),
    [animateEdges, edges],
  );

  return (
    <div {...stylex.props(s.wrapper)}>
      <ReactFlow<CanvasNode, CanvasEdge>
        nodes={nodes}
        edges={renderedEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={onPaneReady}
        onPaneClick={onPaneClick}
        isValidConnection={isValidConnection}
        fitView
        fitViewOptions={fitViewOptions ?? { padding: 0.18, maxZoom: 0.98 }}
        minZoom={0.35}
        maxZoom={1.7}
        colorMode={resolved}
        panOnScroll
        panOnDrag={[1, 2]}
        selectionOnDrag
        multiSelectionKeyCode="Shift"
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          animated: false,
          type: WORKFLOW_EDGE_TYPE,
          style: {
            stroke: tokens.colorAccent,
            strokeWidth: 2,
            opacity: 1,
          },
        }}
        connectionLineStyle={{
          stroke: tokens.colorAccent,
          strokeWidth: 2,
        }}
      >
        {children}
        <Background
          variant={BackgroundVariant.Lines}
          gap={54}
          size={0.65}
          color={tokens.colorGrid}
        />
        <Controls
          className="ns-flow-controls"
          showInteractive={false}
          showFitView={false}
          style={{
            backgroundColor: tokens.colorFlowControls,
            borderColor: tokens.colorBorderStrong,
            borderRadius: "7px",
            overflow: "hidden",
          }}
        />
      </ReactFlow>
    </div>
  );
}

export { addEdge, applyNodeChanges, applyEdgeChanges };
