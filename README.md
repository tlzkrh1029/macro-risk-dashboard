# macro-risk-dashboard

미국 증시와 크립토 노출의 크기를 정하기 위한 매크로 리스크 모니터링 체계입니다.

> 지표 해석 프레임에 관한 정리이며 투자 자문이 아닙니다.

## 문서

| 파일 | 내용 |
|---|---|
| [`docs/v2-design.md`](docs/v2-design.md) | **현재 기준 설계 (v2, 2026-10-14).** 반복해서 볼 문서입니다 |
| [`docs/review-2026-10-14.md`](docs/review-2026-10-14.md) | v1 검토 보고서. 무엇이 왜 바뀌었는지와 데이터 근거 |
| [`docs/v1-design.md`](docs/v1-design.md) | 최초 설계 (v1, 2026-08-19). 비교용으로 보존 |

## 검증 스크립트

`analysis/verify_v1.py`는 검토 보고서에 나온 수치(금리 분해, 백분위, 프록시 오염도, BTC 상관, 복합 경보 백테스트)를 다시 계산합니다. API 키 없이 연준, 뉴욕 연은, Yahoo Finance 데이터를 내려받습니다.

```bash
pip install pandas numpy scipy xlrd requests
python analysis/verify_v1.py            # 데이터를 analysis/.cache/에 내려받고 계산
python analysis/verify_v1.py --offline  # 내려받은 데이터로 다시 계산
```
