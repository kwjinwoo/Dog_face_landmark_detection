# 광운대학교 산학연계 SW 프로젝트
AZZIT 팀 : 정진우, 김종민, 이주완, 최지욱   
지도교수 : 이상민 교수님😁   
연계 회사 : (주) 꼬리
- 프로젝트 개요    
  ![overview](./images/project_overview.png)
  * 모델   
    - mobilenetV2 기반의 Landmark Detection model
    - Imagenet에 Pretrained된 weight 사용
    - 데이터 셋 : CU-Dataset 사용  
      - ![dataset]()
    - 경량화 : TF-Lite의 양자화 기법 사용
      - ![lite]()
  * 애플리케이션
    * android 애플리케이션 개발
    * adroid studio 사용
- 모델 학습
  -  CU-Dataset을 이용한 모델 학습
  -  8개의 Landmark point를 regression
  -  성능 향상을 위해 종 분류를 추가해, multi-task learning 기법 사용
     - ![model structure]()
  - 학습 결과
    - ![loss]()
- 애플리케이션 구현
  - 각도 조절
    - ![angle]()
  - 실제 화면
    - ![app]()