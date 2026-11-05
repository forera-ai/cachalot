#!/bin/bash
# Two arms in sequence, budget 2 then exact (docs/glm-image-ocr/FORM-RECORD.md), with a settle between. About 2 hours.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
docs/glm-image-ocr/arm-form.sh A budget2
benchmarks/settle.sh --budget-gib 40 --timeout 600 || echo "settle did not finish, continuing"
docs/glm-image-ocr/arm-form.sh B exact
echo "all arms done $(date +%T)"
