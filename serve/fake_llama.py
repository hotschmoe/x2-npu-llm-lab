# Tiny fake llama-server for CPU-only tests of decisions_shim (tokenize, apply-template, completion, one passthrough route).
import json, math, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _send(self,o,code=200):
        b=json.dumps(o).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({'status':'ok'})
    def do_POST(self):
        b=json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if self.path=='/tokenize': self._send({'tokens':[1000+ord(b['content'][0])]})
        elif self.path=='/apply-template': self._send({'prompt':'\n'.join(m['content'] for m in b['messages'])+'\n<think>\n\n</think>\n\n'})
        elif self.path=='/completion':
            p=b['prompt']; ids=[1000+ord(c) for c in 'ABCyYnN01234']
            top=[{'id':i,'token':chr(i-1000),'logprob':-1.0-0.3*k} for k,i in enumerate(ids)]
            if 'urgent' in p.lower() or 'today' in p.lower(): top[3]['logprob']=-0.1
            self._send({'completion_probabilities':[{'id':1,'top_logprobs':top}],'timings':{'prompt_n':5,'cache_n':50,'prompt_ms':3.0}})
        else: self._send({'echo':self.path})
ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
