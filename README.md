## 또 DB 연결이 안돼요

```bash
git submodule update --init --recursive`
```

```bash
pip install sshtunnel python-dotenv "paramiko<4.0.0"
```

그래도 안 되면 git push 후 조성혁 호출

## 주의사항

`backend-secret` 폴더 내용, .gitmodules 절대 건들지 마세요 건들면 망함