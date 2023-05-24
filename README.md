# Sentiment Analysis
In this project, we introduce models for the sentiment analysis task. Our solutions tackle sentiment analysis on three levels:
1. Document-level sentiment analysis
2. Sentence-level sentiment analysis
3. Aspect-based sentiment analysis

# Project Setup
Our models use external pretrained vectors and embeddings models. Since these models are large files, we provide install links so users can download these models themselves.
Here is the complete step-by-step procedure:
1. create folder in project root: [`./pretrained_vectors`](./pretrained_vectors)
2. download fasttext pretrained embeddings from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz and save them as [`./pretrained_vectors/cc.en.300.vec`](./pretrained_vectors/cc.en.300.vec)
3. download Transformer-based USE embeddings from https://tfhub.dev/google/universal-sentence-encoder-large/5 and save it as [`./pretrained_vectors/use_transformer`](./pretrained_vectors/use_transformer) 
4. download DAN-based USE embeddings from https://tfhub.dev/google/universal-sentence-encoder/4 and save it as [`./pretrained_vectors/use_dan`](./pretrained_vectors/use_dan)
5. download Sentiment140 dataset from https://drive.google.com/file/d/1vYZWV8RqpDnjgrOkMQHV-eWuHAabtokF/view?usp=share_link and save it as [`./sentence_and_document_level_sa_task/data/sentiment140/data_all_sentiment140.csv`](./sentence_and_document_level_sa_task/data/sentiment140/data_all_sentiment140.csv)

After completing all of these steps, you can run our classification models.