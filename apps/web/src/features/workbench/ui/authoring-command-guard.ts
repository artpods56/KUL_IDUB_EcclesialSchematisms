/**
 * Policy for locally-originated graph commands while local authoring is
 * paused (a persistence operation or execution is in flight).
 */

export interface AuthoringCommandOptions {
  /**
   * Skip the room broadcast. Used when a remote command is replayed locally
   * so it is not sent back to the room.
   */
  syncRoom?: boolean;
  /**
   * The command records the result of a file upload that is still in flight.
   * That upload is the very thing pausing local authoring, but its completion
   * must still commit — otherwise the node stays busy and the whole graph
   * stays locked. Only the upload's own completion may set this; unrelated
   * commands must never use it to bypass the pause.
   */
  isUploadCompletion?: boolean;
}

/**
 * Whether a locally-originated command must be rejected because local
 * authoring is paused. Room replays and file-upload completions are exempt:
 * the former are not new authoring, and the latter is the completion of the
 * operation that is pausing authoring.
 */
export function shouldBlockAuthoringCommand(
  localAuthoringEnabled: boolean,
  options?: AuthoringCommandOptions,
): boolean {
  if (options?.syncRoom === false) return false;
  if (localAuthoringEnabled) return false;
  return options?.isUploadCompletion !== true;
}