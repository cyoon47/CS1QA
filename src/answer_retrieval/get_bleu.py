from nltk.translate.bleu_score import sentence_bleu
import jsonlines
import json

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for r in reader:
            data.append(r)
    
    return data

def main():
    corpus_data = read_jsonl('./corpus_original.jsonl')
    with open('./pred_doc_ids.json', 'r') as jsonfile:
        pred_data = json.load(jsonfile)

    bleu_scores = []
    class_names = ['code_understanding', 'logical', 'error', 'usage', 'algorithm', 'task', 'reasoning', 'code_explain', 'variable']
    type_bleu = [[] for i in range(len(class_names))]
    type_bleu_2 = [[] for i in range(len(class_names))]
    for query_index, pred_text in pred_data:
        query_index = int(query_index)
        query_answer = corpus_data[query_index]['answer']
        for corpus_ex in corpus_data:
            if corpus_ex['question'] == pred_text:
                pred_answer = corpus_ex['answer']
                pred_type = corpus_ex['questionType']

        reference = [query_answer.split()]
        candidate = pred_answer.split()

        score = sentence_bleu(reference, candidate, weights=(1,0,0,0))
        bleu_scores.append(score)
        type_bleu[class_names.index(corpus_data[query_index]['questionType'])].append(score)
        type_bleu_2[class_names.index(pred_type)].append(score)

        # if score > 0.3:
        #     print(query_answer + "\n------------------")
        #     print(pred_answer + "\n------------------")
        #     print(corpus_data[query_index]['questionType'], pred_type)
        #     print()

    # print(bleu_scores)
    # print(len(bleu_scores))

    print("Mean BLEU Score: %.4f" % (sum(bleu_scores)/len(bleu_scores)))

    st_qns = []
    for x in type_bleu_2[:-3]:
        st_qns.extend(x)
    
    ta_qns = []
    for x in type_bleu_2[-3:]:
        ta_qns.extend(x)
    

    avg_bleus_st = sum(st_qns)/len(st_qns)
    avg_bleus_ta = sum(ta_qns)/len(ta_qns)
    print(avg_bleus_st)
    print(avg_bleus_ta)

if __name__ == '__main__':
    main()
