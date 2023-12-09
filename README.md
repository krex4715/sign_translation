# Hand Sign Recognition Toy Project

<img src="./img/고급딥러닝.gif" width="100%">
<img src="./img/delete.gif" width="100%">
<img src="./img/공백.gif" width="100%">
<img src="./img/쌍자음.gif" width="100%">





## Instructions


<img src="./img/ref.jpeg" width="50%">   

* 양손이 무조건 카메라에 위치해있어야 동작
* 왼손은 파란색, 오른손은 빨간색


#### 왼손 : 자음/모음/띄어쓰기/지우기/쌍자음 을 *'선택'*
- 엄지와 검지를 붙였다가 떼는 순간 오른손 *'자음입력'*
- 엄지와 중지를 붙였다가 떼는 순간 오른손 *'모음입력'*
- 엄지와 약지를 붙였다가 떼는 순간 *'띄어쓰기'*
- 엄지와 새끼를 붙였다가 떼는 순간 *'지우기'*
- 따봉표시는 ('ㅎ' 자음) *'쌍자음'*


#### 오른손 : 자음/모음/띄어쓰기/지우기/쌍자음 을 *'입력'*
- 자음, 모음에 해당하는 손가락 모양 입력





※ ㅗㅐ, ㅗㅏ, ㅜㅔ, ㅜㅓ 는 두개의 모음조합을 입력해야함   
※ 아직 ㄺ, ㄻ, ㄼ, ㄽ, ㄾ, ㄿ, ㅀ, ㅄ 는 불가능


***
## Dependency
- [mediapipe](https://developers.google.com/mediapipe)   
    ```pip install mediapipe```
- [opencv-python](https://pypi.org/project/opencv-python/)   
    ```pip install opencv-python```
- [pytorch](https://pytorch.org/)
- [hangul_utils](https://github.com/kaniblu/hangul-utils)   
