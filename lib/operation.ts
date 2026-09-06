/** Owns async results; replacing an operation makes all older results inert. */
export class OperationGate {
  private controller?: AbortController;
  private generation = 0;
  busy = false;

  start() {
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const generation = ++this.generation;
    this.busy = true;
    return { signal: controller.signal, current: () => generation === this.generation && !controller.signal.aborted };
  }

  finish(operation: { current: () => boolean }) {
    if (operation.current()) this.busy = false;
  }

  cancel() {
    this.controller?.abort();
    this.generation += 1;
    this.busy = false;
  }
}
