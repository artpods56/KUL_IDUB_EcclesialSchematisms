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

import { tokens } from "@/lib/stylex/tokens.stylex";
import { connectionIsValid } from "./handles";
import PrototypeNode from "./nodes/PrototypeNode";
import { PROTOTYPE_NODE_TYPE, type PrototypeNodeData } from "./types";

type PrototypeFlowNode = Node<PrototypeNodeData, typeof PROTOTYPE_NODE_TYPE>;

export const nodeTypes: NodeTypes = {
  [PROTOTYPE_NODE_TYPE]: PrototypeNode,
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
  nodes: PrototypeFlowNode[];
  edges: FlowEdge[];
  onNodesChange: OnNodesChange<PrototypeFlowNode>;
  onEdgesChange: OnEdgesChange<FlowEdge>;
  onConnect: OnConnect;
  isValidConnection?: IsValidConnection<FlowEdge>;
  onPaneReady?: (
    instance: ReactFlowInstance<PrototypeFlowNode, FlowEdge>,
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
  const renderedEdges = React.useMemo(
    () => edges.map((edge) => ({ ...edge, animated: animateEdges })),
    [animateEdges, edges],
  );

  return (
    <div {...stylex.props(s.wrapper)}>
      <ReactFlow<PrototypeFlowNode, FlowEdge>
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
        colorMode="dark"
        panOnScroll
        selectionOnDrag
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          animated: false,
          type: "default",
          style: {
            stroke: tokens.colorAccent,
            strokeWidth: 4,
            opacity: 1,
          },
        }}
        connectionLineStyle={{
          stroke: tokens.colorAccent,
          strokeWidth: 4,
        }}
      >
        <Background
          variant={BackgroundVariant.Lines}
          gap={18}
          size={0.65}
          color="rgba(255,255,255,0.035)"
        />
        <Controls
          className="ns-flow-controls"
          showInteractive={false}
          showFitView={false}
          style={{
            backgroundColor: "rgba(29, 31, 35, 0.92)",
            borderColor: tokens.colorBorderStrong,
            borderRadius: "7px",
            overflow: "hidden",
            boxShadow: "0 8px 22px rgba(0,0,0,0.28)",
          }}
        />
      </ReactFlow>
    </div>
  );
}

export { addEdge, applyNodeChanges, applyEdgeChanges };
