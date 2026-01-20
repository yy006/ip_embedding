from tqdm import tqdm

def _w2v(data):
    w2v ={}
    v2w = {}
    fla_d = data.flatten()
    for i in tqdm(fla_d):
        if i not in w2v:
            w2v[i] = len(w2v)
            v2w[len(w2v)-1] = i

    return w2v,v2w

def _corpus(data,w2v):
    corpus = [[w2v[w] for w in ww]  for ww in tqdm(data)]
    return corpus

def _frequency(data):
    freq = {}
    fla_d = data.flatten()
    for w in tqdm(fla_d):
        if w not in freq:
            freq[w] = 0
        freq[w] += 1
    return freq
def _data_loader(corpus,batch_size):

    def func(x):
        if len(x) >= 4:
            return [[x[0],x[1]],[x[0],x[2]],[x[0],x[3]],[x[0], x[4]],[x[2],x[1]],[x[3],x[1]]]
        else:
            # xの長さが4未満の場合の処理を記述します。
            # 例えば、エラーメッセージを表示するなど。
            print(f"Error: The length of x is less than 4. x: {x}")
            return None

    def flatten(nested_list):
        return [e for inner_list in nested_list for e in inner_list]


    l = [func(x) for x in tqdm(corpus)]
    del corpus
    return l
    #l = pd.DataFrame(flatten(l)).to_numpy()
    #batch = [l[batch_size*(i-1):batch_size*i] for i in tqdm(range(1,int(len(l)/batch_size)))]
    #return batch