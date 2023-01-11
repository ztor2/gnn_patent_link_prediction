### 그래프 신경망을 이용한 국방과학기술 융합 예측 양상(Prediction of Defense Science and Technology Convergence using Graph Neural Networks)

- http://www.riss.kr/link?id=A108282300


<br>

**codes:** 실험에 사용된 jupyter notebook 코드 파일 및 모듈 파일을 포함합니다.<br>
**data:** 실험에 사용된 데이터 파일을 포함합니다.<br>
**results:** 실험 결과값 파일들을 포함합니다.<br><br>


#### <code 폴더>
- **construct_original_graph.ipynb:** 원 데이터를 모두 사용하여 모든 정보를 포함하는 IPC 네트워크 그래프를 구축합니다.
- **construct_reduced_graph.ipynb:** 모델별로 링크 예측 검증이 가능하도록, 2019-2020년 데이터에는 존재하나 ~2018년까지의 데이터에는 나타나지 않는 IPC를 제거하고 IPC 네트워크 그래프(reduced graph)를 구축합니다(훈련셋에 존재하지 않는 데이터를 테스트 데이터에서 예측할 수 없으므로).
- **validation_baseline_sc&dw.ipynb:** Reduced graph를 이용하여 spectral clustering 및 DeepWalk 모델의 링크 예측 성능을 검증합니다.
- **validation_baseline_centrality_node_emb.ipynb:** Reduced graph를 이용하여 중심성 기반 링크 예측 모델의 성능을 검증합니다.
- **validation_baseline_topological_edge_score.ipynb:** Reduced graph를 이용하여 네트워크 위상 지표 기반 링크 예측 모델의 성능을 검증합니다.
- **validation_gae.ipynb:** Reduced graph를 이용하여 그래프 오토인코더 모델의 링크 예측 성능을 검증합니다.
- **prediction_gae.ipynb:** Original graph를 모두 훈련셋으로 활용하여 미래의(2020년 이후) 링크를 예측합니다.
- **results_analysis.ipynb:** 링크 예측 결과를 실제 엣지로 추가하고, 기존 네트워크와의 양상 변화를 중심성을 중심으로 분석합니다.
- **gae 폴더:** 그래프 오토인코더 모델 구축에 필요한 각종 함수를 포함합니다.
- **sc_dw 폴더:** spectral clustering 및 DeepWalk 모델 구축에 필요한 각종 함수를 포함합니다.<br><br>


#### <data 폴더>
- **add_patent.xlsx:** 특허별 IPC를 나타낸 raw 데이터. (excel 파일)
- **idx2nodes.pkl / nodes2idx.pkl:** 문자열 타입인 IPC를 일반 index에 대응시키는 dictionary 파일. 네트워크 데이터가 오토인코더 모델에 들어갈 때 문자열 타입인 노드명이 소실되므로 추후 결과 분석에 필요하다. (pickle 타입)
- **original.graph:** 원 데이터를 모두 사용하여 구축한 IPC 네트워크 그래프. (json 타입)
- **reduced_train.graph:** Reduced graph의 train 그래프. (json 타입)
- **reduced_val.graph:** Reduced graph의 validation 그래프. (json 타입)
- **val_edges.pkl / val_non_edges.pkl:** validation 그래프의 edge/non-edge를 나타낸 리스트. 모델에 들어가는 데이터 형태를 맞추어주기 위해 필요. (pickle 타입)
- **val_edges_name.pkl / val_non_edges_name.pkl:** val_edges.pkl / val_non_edges.pkl 를 원 노드명인 IPC로 나타낸 리스트. (실제로는 실험에 사용되지 않음) (pickle 타입)<br><br>


#### <results 폴더>
- **SC_results_ADD_patent.json:** spectral clustering 검증 실험 결과. (논문에는 수록되지 않음)
- **DW_results_ADD_patent.json:** DeepWalk 검증 실험 결과. (논문에는 수록되지 않음)
- **node_emb_results_ADD_patent.json:** 중심성 기반 링크 예측 모델 검증 실험 결과.
- **topo_edge_score_results_ADD_patent.json:** 위상적 특징 기반 링크 예측 모델 검증 실험 결과.
- **GAE_results_ADD_patent.json:** 그래프 오토인코더 링크 예측 모델 검증 실험 결과.


