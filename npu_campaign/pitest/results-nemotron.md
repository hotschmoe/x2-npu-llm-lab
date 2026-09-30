# PI tool-calling suite: nemotron

Model: nemotron  |  server: see logs

PI 0.80.2 (`--mode json -p --no-session`, tools read/bash/edit/write, thinking level off, maxTokens 4096); sampler is server-side (see server cmd in logs/server-nemotron.log).

| case | pass | tool calls (avg) | bash exit!=0 | bad tool calls | leaks | ctx tokens (max) | prefill s (sum) | prefill tok/s | decode tok/s | wall s (avg) |
|---|---|---|---|---|---|---|---|---|---|---|
| create_file | 3/3 | 1.3 | 0 | 0 | 0 | 1925 | 7.2 | 715 | 27.3 | 8 |
| read_answer | 3/3 | 2.0 | 0 | 0 | 0 | 2355 | 8.3 | 668 | 26.8 | 18 |
| shell_cmd | 2/3 | 1.3 | 0 | 0 | 0 | 2152 | 7.2 | 715 | 26.7 | 18 |
| py_func_test | 3/3 | 3.0 | 0 | 0 | 0 | 4248 | 9.3 | 620 | 26.3 | 86 |
| edit_file | 3/3 | 3.0 | 0 | 0 | 0 | 2533 | 9.0 | 613 | 26.9 | 22 |
| find_count | 3/3 | 4.3 | 0 | 0 | 0 | 2452 | 9.7 | 577 | 27.1 | 21 |
| parallel_read | 3/3 | 5.3 | 0 | 0 | 0 | 2441 | 9.0 | 616 | 26.9 | 20 |
| long_result | 3/3 | 1.3 | 0 | 0 | 0 | 6765 | 22.1 | 856 | 25.8 | 21 |
| no_tool | 3/3 | 0.0 | 0 | 0 | 0 | 1811 | 6.2 | 814 | 26.9 | 7 |
| fix_bug | 3/3 | 5.0 | 1 | 0 | 0 | 3449 | 10.9 | 621 | 26.3 | 46 |
| rename_multi | 3/3 | 14.7 | 3 | 1 | 0 | 6653 | 14.9 | 451 | 26.4 | 120 |

**Overall: 32/33 passed.**

## Runs

| run | result | tools | calls | tool errs (bash nonzero + bad) | parallel | out tok | prefill tok/ms | gen tok/ms | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| create_file-r1 | PASS | write,read | 2 | 0 (0+0) | 1 | 205 | 1723/2558 | 205/7519 | 11 |  |
| read_answer-r1 | PASS | bash,read | 2 | 0 (0+0) | 1 | 314 | 1857/2794 | 314/11588 | 15 |  |
| shell_cmd-r1 | FAIL | bash | 1 | 0 (0+0) | 1 | 401 | 1694/2296 | 401/14961 | 18 | answer='' want JPUBUYFW tool_calls=1 |
| py_func_test-r1 | PASS | write,write,bash | 3 | 0 (0+0) | 1 | 2346 | 1906/3055 | 2346/88804 | 93 |  |
| edit_file-r1 | PASS | read,edit,read | 3 | 0 (0+0) | 1 | 292 | 1847/2995 | 292/10702 | 14 |  |
| find_count-r1 | PASS | bash,bash,bash,write | 4 | 0 (0+0) | 1 | 469 | 1845/3089 | 469/17116 | 21 |  |
| parallel_read-r1 | PASS | read,read,read,read,write | 5 | 0 (0+0) | 1 | 447 | 1808/3264 | 447/16354 | 20 |  |
| long_result-r1 | PASS | bash,read | 2 | 0 (0+0) | 1 | 368 | 6400/7643 | 368/14000 | 22 |  |
| no_tool-r1 | PASS |  | 0 | 0 (0+0) | 0 | 136 | 1676/2058 | 136/5011 | 8 |  |
| fix_bug-r1 | PASS | read,read,bash,edit,bash | 5 | 1 (1+0) | 2 | 928 | 2231/3641 | 928/34938 | 39 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "Traceback (most recent call last):\r\n  File \"C:\\Users\\HOTSCH~1\\AppData\\Local\\Temp\\pitest-fix_bug-7_zz54ap\\test_inventory.py\", line 3, i |
| rename_multi-r1 | PASS | bash,bash,read,read,read,read,edit,read,read,read,read,edit,edit,edit,edit,bash,bash,bash | 18 | 2 (1+1) | 4 | 4257 | 2408/5791 | 4257/162146 | 169 |  ERR=tool_error:edit:{"content": [{"type": "text", "text": "Could not find edits[1] in ./pricing.py. The oldText must match exactly including all whitespace and newlines."}], "details": {}}; tool_error:bas |
| create_file-r2 | PASS | write | 1 | 0 (0+0) | 1 | 104 | 1696/2303 | 104/3815 | 7 |  |
| read_answer-r2 | PASS | bash,read | 2 | 0 (0+0) | 1 | 499 | 1859/2765 | 499/18625 | 22 |  |
| shell_cmd-r2 | PASS | bash,bash | 2 | 0 (0+0) | 1 | 407 | 1748/2598 | 407/15151 | 18 |  |
| py_func_test-r2 | PASS | write,write,bash | 3 | 0 (0+0) | 1 | 2170 | 1952/3149 | 2170/81689 | 85 |  |
| edit_file-r2 | PASS | read,edit,read | 3 | 0 (0+0) | 1 | 479 | 1847/2998 | 479/17492 | 21 |  |
| find_count-r2 | PASS | bash,bash,bash,bash,write | 5 | 0 (0+0) | 1 | 562 | 1896/3420 | 562/20779 | 25 |  |
| parallel_read-r2 | PASS | read,read,read,read,write | 5 | 0 (0+0) | 4 | 426 | 1769/2661 | 426/15914 | 19 |  |
| long_result-r2 | PASS | read | 1 | 0 (0+0) | 1 | 281 | 6235/7178 | 281/10899 | 19 |  |
| no_tool-r2 | PASS |  | 0 | 0 (0+0) | 0 | 94 | 1675/2051 | 94/3473 | 6 |  |
| fix_bug-r2 | PASS | bash,read,read,edit,bash | 5 | 0 (0+0) | 2 | 1195 | 2259/3600 | 1195/45187 | 49 |  |
| rename_multi-r2 | PASS | bash,bash,read,read,read,read,edit,edit,edit,edit,bash,bash | 12 | 1 (1+0) | 4 | 2650 | 2089/4052 | 2650/100346 | 106 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "(no output)\n\nCommand exited with code 1"}], "details": {}} |
| create_file-r3 | PASS | write | 1 | 0 (0+0) | 1 | 111 | 1696/2289 | 111/4030 | 7 |  |
| read_answer-r3 | PASS | bash,read | 2 | 0 (0+0) | 1 | 237 | 1858/2782 | 237/8903 | 17 |  |
| shell_cmd-r3 | PASS | bash | 1 | 0 (0+0) | 1 | 411 | 1695/2292 | 411/15523 | 18 |  |
| py_func_test-r3 | PASS | write,write,bash | 3 | 0 (0+0) | 1 | 1948 | 1909/3104 | 1948/74984 | 79 |  |
| edit_file-r3 | PASS | read,edit,read | 3 | 0 (0+0) | 1 | 689 | 1848/3047 | 689/26051 | 30 |  |
| find_count-r3 | PASS | bash,bash,write,read | 4 | 0 (0+0) | 1 | 334 | 1838/3166 | 334/12459 | 16 |  |
| parallel_read-r3 | PASS | bash,read,read,read,read,write | 6 | 0 (0+0) | 4 | 445 | 2000/3123 | 445/16784 | 20 |  |
| long_result-r3 | PASS | read | 1 | 0 (0+0) | 1 | 397 | 6267/7251 | 397/15618 | 23 |  |
| no_tool-r3 | PASS |  | 0 | 0 (0+0) | 0 | 135 | 1677/2070 | 135/5067 | 8 |  |
| fix_bug-r3 | PASS | bash,read,read,edit,bash | 5 | 0 (0+0) | 2 | 1160 | 2259/3627 | 1160/44500 | 49 |  |
| rename_multi-r3 | PASS | bash,bash,bash,read,read,read,read,read,edit,edit,edit,edit,bash,bash | 14 | 1 (1+0) | 5 | 2091 | 2242/5094 | 2091/78766 | 85 |  ERR=tool_error:bash:{"content": [{"type": "text", "text": "(no output)\n\nCommand exited with code 1"}], "details": {}} |
