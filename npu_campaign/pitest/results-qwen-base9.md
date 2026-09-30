# PI tool-calling suite: qwen

Model: qwen  |  server: llama-server pkg-wos -c 16384 -b 2048 -ub 1024 --jinja --reasoning auto --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0

PI 0.80.2 (`--mode json -p --no-session`, tools read/bash/edit/write, thinking level off, maxTokens 4096); sampler is server-side (see server cmd in logs/server-qwen.log).

| case | pass | tool calls (avg) | failed tool calls | leaks | ctx tokens (max) | prefill s (sum) | prefill tok/s | decode tok/s | wall s (avg) |
|---|---|---|---|---|---|---|---|---|---|
| create_file | 3/3 | 1.0 | 0 | 0 | 1785 | 6.0 | 602 | 26.1 | 8 |
| read_answer | 3/3 | 1.0 | 0 | 0 | 1788 | 5.2 | 589 | 26.2 | 7 |
| shell_cmd | 3/3 | 1.0 | 0 | 0 | 1731 | 5.0 | 592 | 26.4 | 6 |
| py_func_test | 3/3 | 4.7 | 0 | 0 | 4770 | 9.9 | 441 | 25.4 | 95 |
| edit_file | 3/3 | 3.7 | 0 | 0 | 2143 | 7.4 | 450 | 26.0 | 16 |
| find_count | 3/3 | 3.3 | 0 | 0 | 2049 | 6.3 | 494 | 25.8 | 15 |
| parallel_read | 3/3 | 5.0 | 0 | 0 | 2214 | 6.0 | 522 | 25.5 | 18 |
| long_result | 3/3 | 1.0 | 0 | 0 | 6843 | 23.5 | 771 | 25.5 | 13 |
| no_tool | 3/3 | 0.0 | 0 | 0 | 1690 | 4.3 | 662 | 26.3 | 5 |

**Overall: 27/27 passed.**

## Runs

| run | result | tools | calls | fail | parallel | out tok | prefill tok/ms | gen tok/ms | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| create_file-r1 | PASS | write | 1 | 0 | 1 | 132 | 1645/2620 | 132/5052 | 8 |  |
| read_answer-r1 | PASS | read | 1 | 0 | 1 | 104 | 1090/1824 | 104/3960 | 6 |  |
| shell_cmd-r1 | PASS | bash | 1 | 0 | 1 | 89 | 1050/1759 | 89/3358 | 6 |  |
| py_func_test-r1 | PASS | write,write,bash | 3 | 0 | 1 | 1937 | 1317/2897 | 1937/76004 | 79 |  |
| edit_file-r1 | PASS | bash,read,edit | 3 | 0 | 1 | 311 | 1066/2227 | 311/11998 | 15 |  |
| find_count-r1 | PASS | bash,bash,write | 3 | 0 | 1 | 334 | 1059/2183 | 334/12922 | 16 |  |
| parallel_read-r1 | PASS | read,read,read,read,write | 5 | 0 | 4 | 325 | 1030/1957 | 325/12597 | 15 |  |
| long_result-r1 | PASS | read | 1 | 0 | 1 | 110 | 6038/7808 | 110/4274 | 13 |  |
| no_tool-r1 | PASS |  | 0 | 0 | 0 | 93 | 938/1411 | 93/3575 | 5 |  |
| create_file-r2 | PASS | write | 1 | 0 | 1 | 142 | 983/1700 | 142/5452 | 8 |  |
| read_answer-r2 | PASS | read | 1 | 0 | 1 | 103 | 996/1707 | 103/3937 | 6 |  |
| shell_cmd-r2 | PASS | bash | 1 | 0 | 1 | 114 | 959/1629 | 114/4352 | 7 |  |
| py_func_test-r2 | PASS | write,write,bash,edit,edit,bash,read | 7 | 0 | 2 | 2339 | 1778/4129 | 2339/92465 | 97 |  |
| edit_file-r2 | PASS | bash,read,edit,read | 4 | 0 | 1 | 354 | 1130/2555 | 354/13692 | 17 |  |
| find_count-r2 | PASS | bash,bash,write,read | 4 | 0 | 2 | 319 | 1039/2135 | 319/12364 | 15 |  |
| parallel_read-r2 | PASS | read,read,read,read,write | 5 | 0 | 4 | 334 | 1029/1941 | 334/12965 | 15 |  |
| long_result-r2 | PASS | read | 1 | 0 | 1 | 131 | 5998/7761 | 131/5143 | 13 |  |
| no_tool-r2 | PASS |  | 0 | 0 | 0 | 79 | 940/1412 | 79/3031 | 5 |  |
| create_file-r3 | PASS | write | 1 | 0 | 1 | 142 | 985/1683 | 142/5455 | 8 |  |
| read_answer-r3 | PASS | read | 1 | 0 | 1 | 133 | 997/1707 | 133/5069 | 7 |  |
| shell_cmd-r3 | PASS | bash | 1 | 0 | 1 | 97 | 958/1626 | 97/3667 | 6 |  |
| py_func_test-r3 | PASS | write,edit,write,bash | 4 | 0 | 1 | 2680 | 1285/2916 | 2680/105893 | 109 |  |
| edit_file-r3 | PASS | bash,read,edit,read | 4 | 0 | 1 | 359 | 1129/2601 | 359/13620 | 17 |  |
| find_count-r3 | PASS | bash,bash,write | 3 | 0 | 2 | 306 | 1017/1983 | 306/11852 | 15 |  |
| parallel_read-r3 | PASS | read,read,read,read,write | 5 | 0 | 4 | 500 | 1057/2076 | 500/19846 | 23 |  |
| long_result-r3 | PASS | read | 1 | 0 | 1 | 146 | 6039/7882 | 146/5735 | 14 |  |
| no_tool-r3 | PASS |  | 0 | 0 | 0 | 84 | 939/1430 | 84/3119 | 5 |  |
