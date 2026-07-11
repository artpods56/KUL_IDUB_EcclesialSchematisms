"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  type Edge,
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
import WorkflowNodeCard from "./nodes/WorkflowNode";
import { WORKFLOW_NODE_TYPE, type WorkflowNodeData } from "./types";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;

export const nodeTypes: NodeTypes = {
  [WORKFLOW_NODE_TYPE]: WorkflowNodeCard,
};

const s = stylex.create({
  wrapper: {
    position: "relative",
    width: "100%",
    height: "100%",
    backgroundColor: tokens.colorBg,
  },
});

export interface WorkflowCanvasProps<FlowEdge extends Edge = Edge> {
  nodes: WorkflowNode[];
  edges: FlowEdge[];
  onNodesChange: OnNodesChange<WorkflowNode>;
  onEdgesChange: OnEdgesChange<FlowEdge>;
  onConnect: OnConnect;
  isValidConnection?: IsValidConnection<FlowEdge>;
  onPaneReady?: (
    instance: ReactFlowInstance<WorkflowNode, FlowEdge>,
  ) => void;
  onPaneClick?: () => void;
  animateEdges?: boolean;
}

export function WorkflowCanvas<FlowEdge extends Edge = Edge>({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  isValidConnection = connectionIsValid,
  onPaneReady,
  onPaneClick,
  animateEdges = false,
}: WorkflowCanvasProps<FlowEdge>) {
  const { resolved } = useTheme();
  const renderedEdges = React.useMemo(
    () => edges.map((edge) => ({ ...edge, animated: animateEdges })),
    [animateEdges, edges],
  );

  return (
    <div {...stylex.props(s.wrapper)}>
      <ReactFlow<WorkflowNode, FlowEdge>
        nodes={nodes}
        edges={renderedEdges}
        nodeTypes={nodeTypes}
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
        selectionOnDrag
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          animated: false,
          type: "default",
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
