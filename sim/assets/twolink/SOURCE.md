# twolinkbot.urdf source

Generated 2026-11-27 from `src/twolink_gazebo/urdf/twolinkbot.urdf.xacro` (md5 b346f8bd90b99a69cbce72a6e133506d) and
`macros.xacro` (md5 a9b6288f15fb606b316de4d07484291e), repo HEAD 705c8fd plus the URDF blanks filled in the working tree.
Re-generate when the xacro changes.

Changes against the xacro output:
- mesh paths `file://.../meshes/X.STL` -> `meshes/X.STL` (copies of `src/twolink_gazebo/meshes/`, identical md5)
- `<gazebo>` and `<ros2_control>` blocks removed, stray text `link2` in link1 removed
- joint1, joint2 `continuous` -> `revolute` with limits q1, q2 [-3.0, 3.0] rad (user 2026-11-27 12:15: one full turn is fine; 0.14 rad margin from ±pi,
  firmware angle reporting across ±pi untested), effort 10, velocity 3 (placeholders)
- inertial origins kept as in the xacro (on -x while the meshes are on +x)
