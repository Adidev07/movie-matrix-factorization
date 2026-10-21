from common import *
def generate(seed=42):
    rng=np.random.default_rng(seed);u=rng.normal(0,.7,(60,3));v=rng.normal(0,.7,(40,3));pairs=np.array([(a,b) for a in range(60) for b in range(40)])
    rng.shuffle(pairs);pairs=pairs[:1500];ratings=np.clip(3+np.sum(u[pairs[:,0]]*v[pairs[:,1]],axis=1)+rng.normal(0,.15,len(pairs)),1,5)
    return pairs,ratings
def fit(pairs,ratings,n_users=60,n_items=40,k=3,epochs=100,lr=.025,reg=.03):
    rng=np.random.default_rng(7);u=rng.normal(0,.15,(n_users,k));v=rng.normal(0,.15,(n_items,k));bu=np.zeros(n_users);bi=np.zeros(n_items);mean=float(ratings.mean())
    for _ in range(epochs):
        for idx in rng.permutation(len(ratings)):
            a,b=pairs[idx];old_u=u[a].copy();err=mean+bu[a]+bi[b]+u[a]@v[b]-ratings[idx]
            u[a]-=lr*(err*v[b]+reg*u[a]);v[b]-=lr*(err*old_u+reg*v[b])
            bu[a]-=lr*(err+reg*bu[a]);bi[b]-=lr*(err+reg*bi[b])
    return dict(u=u,v=v,bu=bu,bi=bi,mean=np.array(mean))
def predict(m,pairs): return np.clip(m['mean']+m['bu'][pairs[:,0]]+m['bi'][pairs[:,1]]+np.sum(m['u'][pairs[:,0]]*m['v'][pairs[:,1]],axis=1),1,5)
def recommend(m,user,seen,k=5):
    candidates=[i for i in range(len(m['v'])) if i not in seen]
    pairs=np.array([(user,i) for i in candidates],dtype=int)
    if not len(pairs): return []
    scores=predict(m,pairs);return [dict(item=int(candidates[i]),predicted_rating=float(scores[i])) for i in np.argsort(-scores,kind='stable')[:k]]
def main():
    pairs,ratings=generate();tr,va,te=np.split(np.arange(len(pairs)),[1000,1250]);best=None;best_error=float('inf');selection=[]
    for epochs in (40,100):
        m=fit(pairs[tr],ratings[tr],epochs=epochs);rmse=float(np.sqrt(np.mean((predict(m,pairs[va])-ratings[va])**2)));selection.append(dict(epochs=epochs,validation_rmse=rmse))
        if rmse<best_error: best=m;best_error=rmse;chosen=epochs
    pred=predict(best,pairs[te]);baseline=np.full(len(te),ratings[tr].mean());hits=[];basehits=[]
    itemmeans=np.array([ratings[tr][pairs[tr,1]==i].mean() if np.any(pairs[tr,1]==i) else ratings[tr].mean() for i in range(40)])
    for user in range(60):
        seen=set(pairs[tr][pairs[tr,0]==user,1].tolist())|set(pairs[va][pairs[va,0]==user,1].tolist())
        relevant=set(pairs[te][(pairs[te,0]==user)&(ratings[te]>=3.5),1].tolist())
        if not relevant: continue
        top={r['item'] for r in recommend(best,user,seen)};hits.append(len(top&relevant)/len(relevant))
        candidates=[i for i in range(40) if i not in seen];bt=set(sorted(candidates,key=lambda i:-itemmeans[i])[:5]);basehits.append(len(bt&relevant)/len(relevant))
    seen=set(pairs[tr][pairs[tr,0]==0,1].tolist())|set(pairs[va][pairs[va,0]==0,1].tolist())
    report=dict(data='60 fictional users, 40 fictional movies, 1500 unique observed ratings; 1000 train, 250 validation, 250 test',selected_epochs=chosen,selection=selection,test_rmse=float(np.sqrt(np.mean((pred-ratings[te])**2))),global_mean_baseline_rmse=float(np.sqrt(np.mean((baseline-ratings[te])**2))),macro_recall_at_5=float(np.mean(hits)) if hits else None,item_mean_baseline_recall_at_5=float(np.mean(basehits)) if basehits else None,ranking_evaluated_users=len(hits),user_0_recommendations=recommend(best,0,seen),caveat='Random warm-start split on synthetic data. Unobserved ratings are unknown relevance, and ranking evaluation is incomplete. No cold-start or temporal claims.')
    out=ROOT/'results';out.mkdir(exist_ok=True);np.savez(out/'model.npz',**best)
    np.savetxt(out/'ratings.csv',np.column_stack([pairs,ratings]),delimiter=',',header='user_id,item_id,rating',comments='')
    save_report(report)
if __name__=='__main__': main()
