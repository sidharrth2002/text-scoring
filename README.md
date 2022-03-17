# Hybrid Deep Neural Networks for Industrial Text Scoring
Sidharrth Nagappan, Hui-Ngo Goh and Amy Hui-Lan Lim

Faculty of Computing and Informatics, Multimedia University

<i>This is the official implementation of our paper in IEA AIE 2022. The corpus of reports is confidential, but we have provided notebooks for benchmark evaluation of our scoring framework on ASAP-AES.</i>

### Abstract
Academic scoring is mainly explored through the pedagogical fields of Automated Essay Scoring (AES) and Short Answer Scoring (SAS), but text scoring in other domains has received limited attention. This paper focuses on industrial text scoring, namely the processing and adherence checking of long annual reports based on regulatory requirements. To lay the foundation for non-academic scoring, a pioneering corpus of annual reports from Malaysian listed companies is scraped, segmented into sections, and domain experts score relevant sections based on adherence. Subsequently, deep neural non-hierarchical attention-based LSTMs, hierarchical attention networks and longformer-based models are refined and evaluated. Since the longformer outperformed LSTM-based models, we embed it into a hybrid scoring framework that employs lexicon and named entity features, with rubric injection via word-level attention, culminating in a Kappa score of 0.956 and 0.811 in both our corpora, respectively. Though scoring is fundamentally subjective, our proposed models show significant results when navigating thin rubric boundaries and handling adversarial responses. As our work proposes a novel industrial text scoring engine, we hope to validate our framework using more official documentation based on a broader range of regulatory practices, both in Malaysia and the securities commissions of other nations.

<img src="updated-framework.png">
<p align="center"><i>Our hybrid proposal consists of blocks A, B and C.</i></p>

#### Block A (Neural Model)

Block A can either be (i) Non-hierarchical LSTM, (ii) Hierarchical LSTM or (iii) Longformer. (i) and (ii) are traditional RNNs, and (iii) is a transformer. We aim to choose the best of the three architectures.

1. <b>Non-hierarchical LSTM</b> - As the naive baseline, this model is based on the architecture of Taghipour. It consists of word tokenization, 300-dimensional pre-trained GLoVE embeddings, a convolutional layer for n-gram level feature extraction, an LSTM layer, pooling of LSTM states via mean-over-time or attention, followed by a 32-cell FCL, and another FCL with either a discrete softmax (classification) or scalar sigmoid (regression) activation. We replicate the same attention layer implementation of Taghipour, which involves the learning of an attention vector signifying the importance of each time step.

2. <b>Hierarchical LSTM</b> - While non-hierarchical LSTMs model text in a linear sequence, hierarchical LSTMs first split the text into sentences, and then words, resulting in a 3-dimensional input vector <i>(batch size, num sentences, num words)</i>. The word and sentence encoders can either be convolutions or LSTMs. Hence, our experiments test permutations of different encoders at both levels.

3. <b>Longformer</b> - To handle sequences of text longer than 512 words, we employ the Longformer model. By introducing a dilated sliding window and combining local and global attention, the Longformer can handle sequence lengths of up to 4096 words, pre-trained on autoregressive language modelling tasks. While the Longformer is not the state-of-the-art, we contend that it stands on the balance between fast computations and performance compared to RoBERTA and BigBird. Our longformer-based model consists of tokenization, Longformer block embeddings (finetuned), dropout and FCL.

#### Block B (NLP Features)

Block B is a module that processes the categorical presence of domain-specific lexicons and the numerical count of selected named entity families. This is based on the understanding that more detailed responses will make mention of relevant n-grams and named entities in their description. We use the Term Frequency-Inverse Document Frequency (TF-IDF) ranking of n-grams in the lemmatized corpus along with Spacy's rule-based PhraseMatcher to select and mark the 30 most meaningful lexicons, while using StanfordNLP's NER tagger to annotate the corpus before counting the frequency of each NER family (organisations, laws, persons) in each response. A Support Vector Classifier is used with Recursive Feature Elimination to identify handcrafted features that most impact the final score. The numerical features are transformed using a quantile normal distribution, and categorical features are one-hot encoded.

#### Block C (Rubric Word-level Attention)
Based on the work of Chen et al. and Wang et al., we compute the attentional similarity between expert-defined keywords (that one can expect a high-scoring response to use) and the response. The careful selection of phrases allows the injection of scoring rubrics into the model.

For a response <img src="svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode" align=middle width=7.87295519999999pt height=14.15524440000002pt/> with <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> words and key phrase <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> with <img src="svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> words, GLoVE word embedding sequences <img src="svgs/db683cdca5d92271dbbbf1d0d957275a.svg?invert_in_darkmode" align=middle width=121.04849129999998pt height=24.65753399999998pt/> and <img src="svgs/e32a06c85ad18320cde736c7cf5257b8.svg?invert_in_darkmode" align=middle width=126.72777149999999pt height=27.91243950000002pt/> are respectively generated.

1. The dot product of the two sequences is computed.

<img src="svgs/3b77bc0239301f9381a318e43ba7e752.svg?invert_in_darkmode" align=middle width=86.7693255pt height=27.91243950000002pt/>

2. Softmax is computed over the rows and columns of the matrix to obtain <img src="svgs/bb29a9f7a3162180ce87baa9ef88360d.svg?invert_in_darkmode" align=middle width=17.84252744999999pt height=27.91243950000002pt/> and <img src="svgs/d1454c73d1f7e48c0bf5f62ef552cd12.svg?invert_in_darkmode" align=middle width=17.03394659999999pt height=21.839370299999988pt/>, where <img src="svgs/bb29a9f7a3162180ce87baa9ef88360d.svg?invert_in_darkmode" align=middle width=17.84252744999999pt height=27.91243950000002pt/> intuitively signifies the attention that the word <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> in the key phrase pays to every word in <img src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.68915409999999pt height=14.15524440000002pt/>.

<img src="svgs/9ad2c9e3cbde8062adc37be40616e145.svg?invert_in_darkmode" align=middle width=424.22195475pt height=27.91243950000002pt/>

3. Attentional vectors are computed based on <img src="svgs/bb29a9f7a3162180ce87baa9ef88360d.svg?invert_in_darkmode" align=middle width=17.84252744999999pt height=27.91243950000002pt/> and <img src="svgs/ca4a4046292218c69b2df64a5b492206.svg?invert_in_darkmode" align=middle width=17.70688094999999pt height=21.839370299999988pt/> using a weighted sum for both key phrase to response and response to key phrase.


<img src="svgs/69ec3bafb85d22f40297b1414a37af4b.svg?invert_in_darkmode" align=middle width=286.51478955pt height=27.91243950000002pt/>

4. A feature vector <img src="svgs/2ba1068e431d1c8c6ff698a6590a6a4b.svg?invert_in_darkmode" align=middle width=72.4600074pt height=24.65753399999998pt/> is output for <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> key elements before being concatenated into an overall word-level attention vector <img src="svgs/84ff826e04cd5f2ba4391db3aefa8f92.svg?invert_in_darkmode" align=middle width=138.88708515pt height=24.65753399999998pt/>.

#### Combining Module for Multimodal Amalgamation
The outputs of blocks B and C are combined with the logits output of the neural block either via an (a) attention sum or (b) an MLP. The attention sum adds the Longformer outputs, categorical and numerical features and the word-level attention feature vector, before the Longformer outputs query the result vector. For example, if <img src="svgs/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/> is the final feature vector, <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> is the weight matrix, <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> is the longformer's text features, <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.11380504999999pt height=14.15524440000002pt/> represents categorical features, <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> represents numerical, and <img src="svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode" align=middle width=12.210846449999991pt height=14.15524440000002pt/> represents attentional features, the total features <img src="svgs/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/> and attention <img src="svgs/c7b563f054f93e8efdaffab95f932e14.svg?invert_in_darkmode" align=middle width=25.175188499999987pt height=14.15524440000002pt/> are:

<img src="svgs/edff484280d63794d59534baddce822f.svg?invert_in_darkmode" align=middle width=352.35256649999997pt height=22.465723500000017pt/>

<img src="svgs/899897c262064b1da5323e063631110d.svg?invert_in_darkmode" align=middle width=340.9405791pt height=37.830091200000005pt/>


#### Acknowledgements
<i>We employed and built on top of Ken Gu's Pytorch implementation of multimodal transformers. Pretrained transformer weights are from Huggingface. The Pytorch implementation of word-level attention is based off Wang et al. (2019)'s Tensorflow implementation.</i>

If you have any queries, please contact us through this <a href="mailto:sidharrth2002@gmail.com">email</a>.

#### References

1. Beltagy, I., Peters, M.E., Cohan, A.: Longformer: The long-document transformer. CoRR abs/2004.05150 (2020)
2. Chen,Q.,Zhu,X.,Ling,Z.,Wei,S.,Jiang,H.:Enhancing and combining sequential and tree LSTM for natural language inference. CoRR abs/1609.06038 (2016)
3. Dasgupta, T., Naskar, A., Saha, R., Dey, L.: Augmenting textual qualitative fea-
tures in deep convolution recurrent neural network for automatic essay scoring. pp.
93–102 (2018)
4. Devlin, J., Chang, M., Lee, K., Toutanova, K.: BERT: pre-training of deep bidirec-
tional transformers for language understanding. CoRR abs/1810.04805 (2018)
5. Dong, F., Zhang, Y., Yang, J.: Attention-based recurrent convolutional neural net-
work for automatic essay scoring. pp. 153–162 (Aug 2017)
6. Gu, K., Budhkar, A.: A package for learning on tabular and text data with trans-
formers. In: Proceedings of the Third Workshop on Multimodal Artificial Intelli-
gence. pp. 69–73. Association for Computational Linguistics (Jun 2021)
7. Kumar, V., Boulanger, D.: Explainable automated essay scoring: Deep learning
really has pedagogical value. Frontiers in Education 5, 186 (2020)
8. Mayfield, E., Black, A.: Should you fine-tune bert for automated essay scoring?
pp. 151–162 (01 2020)
9. OECD: Who are the owners of the world’s listed companies and why should we care?, https://www.oecd.org/corporate/who-are-the-owners-of-the-worlds-listed-companies-and-why-should-we-care.htm
10. Page, E.B.: Project essay grade: Peg. Journal of Educational Technology (2003)
11. Pennington, J., Socher, R., Manning, C.: GloVe: Global vectors for word represen- tation. In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 1532–1543. Association for Computational
Linguistics (Oct 2014). https://doi.org/10.3115/v1/D14-1162
12. Riordan, B., Horbach, A., Cahill, A., Zesch, T., Lee, C.M.: Investigating neural architectures for short answer scoring. pp. 159–168. Association for Computational Linguistics (Sep 2017)
13. Shermis, M.D., Burstein, J.: Automated essay scoring : A cross-disciplinary perspective. In: Proceedings of the 2003 International Conference on Computational
Linguistics. p. 13 (2003)
14. Taghipour, K., Ng, H.T.: A neural approach to automated essay scoring. In: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Pro-
cessing. pp. 1882–1891. Association for Computational Linguistics (11 2016)
15. Uto, M., Xie, Y., Ueno, M.: Neural automated essay scoring incorporating hand- crafted features. In: Proceedings of the 28th International Conference on Computational Linguistics. pp. 6077–6088. International Committee on Computational Linguistics (Dec 2020)
16. Wang, T., Inoue, N., Ouchi, H., Mizumoto, T., Inui, K.: Inject rubrics into short answer grading system. In: Proceedings of the 2nd Workshop on Deep Learning
Approaches for Low-Resource NLP. pp. 175–182 (2019)
17. Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., Hovy, E.: Hierarchical attention networks for document classification. pp. 1480–1489. Association for Computational Linguistics (Jun 2016)
18. Zaheer,M.,Guruganesh,G.,Dubey,A.,Ainslie,J.,Alberti,C.,Ontañón,S.,Pham,
P., Ravula, A., Wang, Q., Yang, L., Ahmed, A.: Big bird: Transformers for longer sequences. CoRR abs/2007.14062 (2020), https://arxiv.org/abs/2007.14062
