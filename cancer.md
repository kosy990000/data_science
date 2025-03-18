//bash

//단어 사전 생성
GloVe/build/vocab_count -min-count 1 -verbose 2 < glove_gene_data.txt > vocab.txt

//공존 행랼 생성
GloVe/build/cooccur -memory 4.0 -vocab-file vocab.txt -verbose 2 < glove_gene_data.txt > cooccurrence.bin

//공존 행렬 섞기
GloVe/build/shuffle -memory 4.0 -verbose 2 < cooccurrence.bin > cooccurrence.shuf.bin

//학습
GloVe/build/glove -save-file vectors -threads 8 -input-file cooccurrence.shuf.bin -x-max 10 -iter 50 -vector-size 100 -binary 2 -vocab-file vocab.txt -verbose 2
// size 100, 반복 50
-vector-size 100: 벡터 크기 100차원 설정
-iter 50: 학습 반복 횟수
-threads 8: CPU 8개 사용 (병렬 처리)
-binary 2: 벡터를 txt와 bin 둘 다 저장
