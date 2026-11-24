"use strict";

// Stored intrinsic XYZ angles describe Rx(x) Ry(y) Rz(z). A rotation about
// the currently displayed local axis composes on the right: Rnew = R Raxis.
function rotateCurrentAxes(angles, axis, degrees) {
  if (
    angles.length !== 3 ||
    !angles.every(Number.isFinite) ||
    !Number.isInteger(axis) ||
    axis < 0 ||
    axis > 2 ||
    !Number.isFinite(degrees)
  )
    throw new Error("旋转角度无效");
  const rad = Math.PI / 180;
  const [x, y, z] = angles.map((v) => v * rad);
  const [sx, sy, sz] = [x, y, z].map(Math.sin);
  const [cx, cy, cz] = [x, y, z].map(Math.cos);
  const rotation = [
    [cy * cz, -cy * sz, sy],
    [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
    [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
  ];
  const c = Math.cos(degrees * rad),
    s = Math.sin(degrees * rad);
  const delta = [
    [
      [1, 0, 0],
      [0, c, -s],
      [0, s, c],
    ],
    [
      [c, 0, s],
      [0, 1, 0],
      [-s, 0, c],
    ],
    [
      [c, -s, 0],
      [s, c, 0],
      [0, 0, 1],
    ],
  ][axis];
  const result = rotation.map((row) =>
    delta[0].map((_, j) => row.reduce((sum, v, k) => sum + v * delta[k][j], 0)),
  );
  const cosY = Math.hypot(result[0][0], result[0][1]);
  // At gimbal lock choose z = 0; the equivalent orientation remains intact.
  const next =
    cosY > 1e-10
      ? [
          Math.atan2(-result[1][2], result[2][2]),
          Math.atan2(result[0][2], cosY),
          Math.atan2(-result[0][1], result[0][0]),
        ]
      : [Math.atan2(result[2][1], result[1][1]), Math.atan2(result[0][2], cosY), 0];
  return next.map((v) => v / rad);
}
