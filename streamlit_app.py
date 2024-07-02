import chromadb 
import ollama 


class AI():
 def __init__(self):
  db = chromadb.PersistentClient()
  self.collection = db.get_or_create_collection("nvidia")

 def query(self, q, top=10):
  res_db = self.collection.query(query_texts=[q])["documents"][0][0:top]
  context = ' '.join(res_db).replace("\n", " ")
  return context

 def respond(self, lst_messages, model="phi3", use_knowledge=False):
  q = lst_messages[-1]["content"]
  context = self.query(q)

  if use_knowledge:
   prompt = "Give the most accurate answer using your knowledge and the following additional information: \n"+context
  else:
   prompt = "Give the most accurate answer using only the following information: \n"+context

  res_ai = ollama.chat(model=model, 
        messages=[{"role":"system", "content":prompt}]+lst_messages,
                      stream=True)
  for res in res_ai:
   chunk = res["message"]["content"]
   app["full_response"] += chunk
   yield chunk


ai = AI()