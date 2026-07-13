"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  type EdgeTypes,
  type IsValidConnection,
  type Node,
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
import { connectionIsValid } from "./handles";
import WorkflowEdgeControl from "./edges/WorkflowEdge";
import WorkflowNodeCard from "./nodes/WorkflowNode";
import {
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  type WorkflowEdge,
  type WorkflowNodeData,
} from "./types";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;

export const nodeTypes: NodeTypes = {
  [WORKFLOW_NODE_TYPE]: WorkflowNodeCard,
};

export const edgeTypes: EdgeTypes = {
  [WORKFLOW_EDGE_TYPE]: WorkflowEdgeControl,
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
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  onNodesChange: OnNodesChange<WorkflowNode>;
  onEdgesChange: OnEdgesChange<WorkflowEdge>;
  onConnect: OnConnect;
  isValidConnection?: IsValidConnection<WorkflowEdge>;
  onPaneReady?: (
    instance: ReactFlowInstance<WorkflowNode, WorkflowEdge>,
  ) => void;
  onPaneClick?: () => void;
  animateEdges?: boolean;
}

export function WorkflowCanvas({
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
    () => edges.map((edge) => ({ ...edge, animated: animateEdges })),
    [animateEdges, edges],
  );

  return (
    <div {...stylex.props(s.wrapper)}>
      <ReactFlow<WorkflowNode, WorkflowEdge>
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
        fitViewOptions={{ padding: 0.18, maxZoom: 0.98 }}
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
