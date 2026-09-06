import test from 'node:test';
import assert from 'node:assert/strict';
import { OperationGate } from '../lib/operation.ts';

test('replacing scene suppresses old results and old finally cannot unlock new work', () => {
  const gate = new OperationGate();
  const old = gate.start();
  const current = gate.start();
  assert.equal(old.current(), false);
  assert.equal(old.signal.aborted, true);
  gate.finish(old);
  assert.equal(gate.busy, true);
  gate.finish(current);
  assert.equal(gate.busy, false);
});

test('cancellation suppresses pending success/error state and releases lock', () => {
  const gate = new OperationGate();
  const operation = gate.start();
  gate.cancel();
  assert.equal(operation.current(), false);
  assert.equal(gate.busy, false);
});
