import test from 'node:test';
import assert from 'node:assert/strict';
import { isValidTimingEdit } from '../lib/timing.ts';

const detected = { start: 1, end: 1.5, detectedStart: 1, detectedEnd: 1.5, minStart: 0.85, maxEnd: 1.65 };

test('repeated edits cannot move a section entirely outside its original speech', () => {
  assert.equal(isValidTimingEdit(detected, 1, 1.6, []), true);
  const edited = { ...detected, end: 1.6 };
  assert.equal(isValidTimingEdit(edited, 1.52, 1.6, []), false);
  assert.equal(isValidTimingEdit(edited, 0.86, 0.96, []), false);
  assert.equal(isValidTimingEdit(edited, 1.45, 1.6, []), true);
});

test('split children retain immutable detection bounds and cannot contain only padding', () => {
  const edited = { ...detected, start: 0.85, end: 1.65 };
  assert.equal(isValidTimingEdit(edited, edited.start, 0.95, []), false);
  const left = { ...edited, end: 1.25 };
  const right = { ...edited, start: 1.25 };
  assert.equal(isValidTimingEdit(left, left.start, left.end, [right]), true);
  assert.equal(isValidTimingEdit(right, right.start, right.end, [left]), true);
  assert.equal(isValidTimingEdit(right, 1.52, 1.64, [left]), false);
});

test('timing edits reject non-finite, short, out-of-bounds and overlapping sections', () => {
  for (const [start, end] of [[NaN, 1.5], [1, Infinity], [1, 1.02], [0.8, 1.5], [1, 1.7]]) {
    assert.equal(isValidTimingEdit(detected, start, end, []), false);
  }
  assert.equal(isValidTimingEdit(detected, 1, 1.5, [{ start: 1.4, end: 2 }]), false);
  assert.equal(isValidTimingEdit(detected, 1.1, 1.14, []), true);
});
