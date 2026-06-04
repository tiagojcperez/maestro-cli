# Maestro CLI — README demo recording.
#
# Generates demo.gif with VHS (https://github.com/charmbracelet/vhs):
#     vhs demo.tape
#
# It uses the engine-free examples/demo_plan.yaml (shell tasks only), so it needs
# NO API keys and runs anywhere. Then reference the result in README.md, e.g.:
#     ![Maestro demo](demo.gif)

Output demo.gif

Set Shell bash
Set FontSize 18
Set Width 1120
Set Height 660
Set Padding 20
Set Theme "Dracula"
Set TypingSpeed 55ms

# A glimpse of the banner + version
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
Sleep 4s

# Deterministic artifacts on disk
Type "ls .maestro-runs/"
Enter
Sleep 2500ms
