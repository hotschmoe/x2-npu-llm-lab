# PI tool-calling suite: gemma

Model: gemma  |  server: llama-server pkg-wos -c 16384 -b 2048 -ub 1024 --jinja --reasoning auto --temp 1.0 --top-p 0.95 --top-k 64

PI 0.80.2 (`--mode json -p --no-session`, tools read/bash/edit/write, thinking level off, maxTokens 4096); sampler is server-side (see server cmd in logs/server-gemma.log).

| case | pass | tool calls (avg) | bash exit!=0 | bad tool calls | leaks | ctx tokens (max) | prefill s (sum) | prefill tok/s | decode tok/s | wall s (avg) |
|---|---|---|---|---|---|---|---|---|---|---|
| create_file | 0/1 | 0.0 | 0 | 0 | 0 | 1389 | 1.9 | 727 | 23.5 | 4 |
| read_answer | 0/1 | 0.0 | 0 | 0 | 0 | 1391 | 1.4 | 742 | 23.5 | 3 |
| shell_cmd | 0/1 | 0.0 | 0 | 0 | 0 | 1466 | 1.5 | 707 | 23.1 | 6 |
| py_func_test | 0/1 | 0.0 | 0 | 0 | 0 | 1513 | 1.6 | 685 | 23.3 | 4 |
| edit_file | 0/1 | 0.0 | 0 | 0 | 0 | 1427 | 1.3 | 714 | 23.3 | 4 |

**Overall: 0/5 passed.**

## Runs

| run | result | tools | calls | tool errs (bash nonzero + bad) | parallel | out tok | prefill tok/ms | gen tok/ms | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| create_file-r1 | FAIL |  | 0 | 0 (0+0) | 0 | 31 | 1359/1869 | 31/1321 | 4 | hello.txt=None |
| read_answer-r1 | FAIL |  | 0 | 0 (0+0) | 0 | 36 | 1026/1383 | 36/1530 | 3 | answer='*.age</h4>age appgeg dis Iif +</html> with2 ا</html> है</b> whang resac</b>� dis ex res<th>ubif<unused76></b>P</b>' want BLUEHHRO 2026-11-24 |
| shell_cmd-r1 | FAIL |  | 0 | 0 (0+0) | 0 | 102 | 1036/1465 | 102/4421 | 6 | answer='ickpp<unused17>age            --ith</h4>�ith�<unused17>�<unused58>��T            E</b> with{el  </html> restherith\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n</h4>{</html>ageEel<unused85>  er��per<unused17>' want JPUBUYFW tool_calls=0 |
| py_func_test-r1 | FAIL |  | 0 | 0 (0+0) | 0 | 51 | 1127/1645 | 51/2191 | 4 | missing solution.py or test_solution.py |
| edit_file-r1 | FAIL |  | 0 | 0 (0+0) | 0 | 52 | 942/1318 | 52/2228 | 4 | config.ini='[client]\nhost = localhost\ntimeout = 30\nretries = 3\n\n[server]\nport = 8080\ntimeout = 30\nworkers = 4\n\n[logging]\nlevel = info\n' |
