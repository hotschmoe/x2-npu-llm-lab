# PI tool-calling suite: qwen-full

Model: qwen  |  server: see logs

PI 0.80.2 (`--mode json -p --no-session`, tools read/bash/edit/write, thinking level off, maxTokens 4096); sampler is server-side (see server cmd in logs/server-qwen-full.log).

| case | pass | tool calls (avg) | bash exit!=0 | bad tool calls | leaks | ctx tokens (max) | prefill s (sum) | prefill tok/s | decode tok/s | wall s (avg) |
|---|---|---|---|---|---|---|---|---|---|---|
| create_file | 3/3 | 1.0 | 0 | 0 | 0 | 1782 | 6.0 | 601 | 27.2 | 9 |
| read_answer | 3/3 | 1.3 | 0 | 0 | 0 | 1843 | 5.4 | 573 | 27.2 | 7 |
| shell_cmd | 3/3 | 1.0 | 0 | 0 | 0 | 1711 | 5.0 | 597 | 27.4 | 6 |
| py_func_test | 3/3 | 4.7 | 1 | 1 | 0 | 4230 | 9.7 | 447 | 26.1 | 76 |
| edit_file | 3/3 | 4.0 | 0 | 0 | 0 | 2200 | 7.6 | 445 | 26.8 | 18 |
| find_count | 3/3 | 2.7 | 0 | 0 | 0 | 2008 | 5.8 | 530 | 26.8 | 13 |
| parallel_read | 3/3 | 5.0 | 0 | 0 | 0 | 2164 | 5.9 | 529 | 26.6 | 16 |
| long_result | 3/3 | 1.7 | 0 | 0 | 0 | 6894 | 23.8 | 762 | 26.2 | 15 |
| no_tool | 3/3 | 0.0 | 0 | 0 | 0 | 1721 | 4.2 | 662 | 26.9 | 6 |
| fix_bug | 3/3 | 5.0 | 3 | 0 | 0 | 2953 | 8.1 | 546 | 26.1 | 30 |
| rename_multi | 3/3 | 12.3 | 1 | 0 | 0 | 3175 | 10.1 | 493 | 26.4 | 44 |

**Overall: 33/33 passed.**

## Runs

| run | result | tools | calls | tool errs (bash nonzero + bad) | parallel | out tok | prefill tok/ms | gen tok/ms | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| create_file-r1 | PASS | write | 1 | 0 (0+0) | 1 | 134 | 1645/2598 | 134/4881 | 12 |  |
| read_answer-r1 | PASS | read | 1 | 0 (0+0) | 1 | 106 | 1091/1805 | 106/3843 | 6 |  |
| shell_cmd-r1 | PASS | bash | 1 | 0 (0+0) | 1 | 96 | 1051/1741 | 96/3468 | 6 |  |
| py_func_test-r1 | PASS | write,write,bash | 3 | 0 (0+0) | 1 | 1656 | 1254/2739 | 1656/62407 | 66 |  |
| edit_file-r1 | PASS | bash,read,edit,read | 4 | 0 (0+0) | 1 | 413 | 1130/2549 | 413/15267 | 19 |  |
| find_count-r1 | PASS | bash,bash,write | 3 | 0 (0+0) | 2 | 313 | 1036/1981 | 313/11597 | 14 |  |
| parallel_read-r1 | PASS | read,read,read,read,write | 5 | 0 (0+0) | 4 | 326 | 1028/1954 | 326/12118 | 15 |  |
| long_result-r1 | PASS | bash,read | 2 | 0 (0+0) | 1 | 179 | 6056/7995 | 179/6752 | 15 |  |
| no_tool-r1 | PASS |  | 0 | 0 (0+0) | 0 | 120 | 936/1413 | 120/4424 | 6 |  |
| fix_bug-r1 | PASS | bash,read,read,edit,bash | 5 | 1 (1+0) | 3 | 617 | 1464/2675 | 617/23330 | 27 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "Traceback (most recent call last):\r\n  File \"C:\\Users\\HOTSCH~1\\AppData\\Local\\Temp\\pitest-fix_bug-hwwsw09f\\test_inventory.py\", line 3, i |
| rename_multi-r1 | PASS | bash,bash,read,read,read,read,read,edit,edit,edit,edit,bash,bash | 13 | 1 (1+0) | 5 | 1110 | 2071/4091 | 1110/41988 | 47 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "(no output)\n\nCommand exited with code 1"}], "details": {}} |
| create_file-r2 | PASS | write | 1 | 0 (0+0) | 1 | 143 | 979/1696 | 143/5237 | 8 |  |
| read_answer-r2 | PASS | read | 1 | 0 (0+0) | 1 | 130 | 994/1685 | 130/4761 | 7 |  |
| shell_cmd-r2 | PASS | bash | 1 | 0 (0+0) | 1 | 96 | 955/1602 | 96/3495 | 6 |  |
| py_func_test-r2 | PASS | write,write,edit,read,edit,bash | 6 | 1 (0+1) | 1 | 1801 | 1774/3837 | 1801/68428 | 73 |  ERR=tool_error:edit:{"content": [{"type": "text", "text": "Could not find the exact text in C:/Users/HOTSCH~1/AppData/Local/Temp/pitest-py_func_test-r51l_aqj/test_solution.py. The old text must match exac |
| edit_file-r2 | PASS | bash,read,edit,read | 4 | 0 (0+0) | 1 | 381 | 1129/2508 | 381/14160 | 17 |  |
| find_count-r2 | PASS | bash,write | 2 | 0 (0+0) | 1 | 248 | 1009/1877 | 248/9185 | 12 |  |
| parallel_read-r2 | PASS | read,read,read,read,write | 5 | 0 (0+0) | 4 | 303 | 1027/1918 | 303/11259 | 14 |  |
| long_result-r2 | PASS | read | 1 | 0 (0+0) | 1 | 169 | 5994/7734 | 169/6514 | 15 |  |
| no_tool-r2 | PASS |  | 0 | 0 (0+0) | 0 | 110 | 936/1412 | 110/4115 | 6 |  |
| fix_bug-r2 | PASS | bash,read,read,edit,bash | 5 | 1 (1+0) | 3 | 705 | 1460/2654 | 705/27139 | 30 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "Traceback (most recent call last):\r\n  File \"C:\\Users\\HOTSCH~1\\AppData\\Local\\Temp\\pitest-fix_bug-mijklip1\\test_inventory.py\", line 3, i |
| rename_multi-r2 | PASS | bash,bash,read,read,read,read,read,edit,edit,edit,edit,bash | 12 | 0 (0+0) | 5 | 995 | 1466/3059 | 995/37806 | 42 |  |
| create_file-r3 | PASS | write | 1 | 0 (0+0) | 1 | 144 | 977/1695 | 144/5370 | 8 |  |
| read_answer-r3 | PASS | bash,read | 2 | 0 (0+0) | 1 | 168 | 1016/1917 | 168/6277 | 9 |  |
| shell_cmd-r3 | PASS | bash | 1 | 0 (0+0) | 1 | 83 | 954/1619 | 83/3061 | 5 |  |
| py_func_test-r3 | PASS | write,write,bash,edit,bash | 5 | 1 (1+0) | 1 | 2193 | 1293/3093 | 2193/85298 | 89 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "FAIL: mixed order \ufffd expected 3720, got 3750\r\nFAIL: multiple same units \ufffd expected 5400, got 6300\r\n\r\n25 passed, 2 failed\r\nSOME T |
| edit_file-r3 | PASS | bash,read,edit,read | 4 | 0 (0+0) | 1 | 358 | 1128/2546 | 358/13568 | 17 |  |
| find_count-r3 | PASS | bash,bash,write | 3 | 0 (0+0) | 2 | 277 | 1013/1910 | 277/10491 | 13 |  |
| parallel_read-r3 | PASS | read,read,read,read,write | 5 | 0 (0+0) | 4 | 450 | 1055/2005 | 450/17149 | 20 |  |
| long_result-r3 | PASS | bash,read | 2 | 0 (0+0) | 1 | 173 | 6056/8025 | 173/6604 | 15 |  |
| no_tool-r3 | PASS |  | 0 | 0 (0+0) | 0 | 124 | 936/1419 | 124/4641 | 7 |  |
| fix_bug-r3 | PASS | bash,read,read,edit,bash | 5 | 1 (1+0) | 3 | 806 | 1489/2750 | 806/31008 | 34 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "Traceback (most recent call last):\r\n  File \"C:\\Users\\HOTSCH~1\\AppData\\Local\\Temp\\pitest-fix_bug-33oyufom\\test_inventory.py\", line 3, i |
| rename_multi-r3 | PASS | bash,bash,read,read,read,read,read,edit,edit,edit,edit,bash | 12 | 0 (0+0) | 5 | 1003 | 1461/2979 | 1003/38139 | 42 |  |
