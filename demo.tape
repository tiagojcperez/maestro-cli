# Maestro CLI — README demo recording (CLI run).
#
# Generates demo.gif with VHS (https://github.com/charmbracelet/vhs):
#     vhs demo.tape
#
# Uses the engine-free examples/demo_plan.yaml (shell tasks only) — NO API keys,
# runs anywhere. Reference it in README.md with:  ![Maestro CLI demo](demo.gif)

Output demo.gif

Set Shell bash
Set FontSize 18
Set Width 1180
Set Height 680
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set TypingSpeed 55ms

Sleep 600ms

# Banner + version
Type "maestro --version"
Enter
Sleep 2s

# Validate the engine-free demo plan
Type "maestro validate examples/demo_plan.yaml"
Enter
Sleep 2500ms

# Run it — a small parallel DAG of shell tasks, no API keys needed
Type "maestro run examples/demo_plan.yaml --max-parallel 3"
Enter
Sleep 4500ms

# Deterministic artifacts on disk
Type "ls .maestro-runs/"
Enter
Sleep 3s
