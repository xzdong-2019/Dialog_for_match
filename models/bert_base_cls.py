import os
import torch
import torch.nn as nn

from models.bert import modeling_bert, configuration_bert

class BERTbase(nn.Module):
  def __init__(self, hparams):
    super(BERTbase, self).__init__()
    self.hparams = hparams

    bert_config = configuration_bert.BertConfig.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                   "%s-config.json" % self.hparams.bert_pretrained),
    )
    self._bert_model = modeling_bert.BertModel.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained, self.hparams.bert_checkpoint_path),
      config=bert_config
    )

    if self.hparams.do_eot and self.hparams.model_type == "bert_base_ft":
      self._bert_model.resize_token_embeddings(self._bert_model.config.vocab_size + 1) # [EOT]
    self.response_span = self.hparams.max_dialog_len//10
    #self.gru_utt = nn.GRU(self.hparams.bert_hidden_dim, self.hparams.bert_hidden_dim)
    #self.gru_res = nn.GRU(self.hparams.bert_hidden_dim, self.hparams.bert_hidden_dim)
    self.gru_all = nn.GRU(self.hparams.bert_hidden_dim, self.hparams.bert_hidden_dim)
    #self.softmax = nn.Softmax(dim=1)

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, 1)
    )
    
  def forward(self, batch):
    bert_outputs, _ = self._bert_model(
      batch["anno_sent"],
      token_type_ids=batch["segment_ids"],
      attention_mask=batch["attention_mask"]
    )
    #import pdb
    #pdb.set_trace()
    uttrance_logits = torch.stack([bert_outputs[:,i*self.response_span,:] for i in range(10)], 1) # bs, bert_output_size
    respones_logits = bert_outputs[:, self.hparams.max_dialog_len,:]
    
    respones_logits = respones_logits(-1, 1, self.hparams.bert_hidden_dim)
    #uttrance_logits = uttrance_logits.view(-1,self.response_span,self.hparams.bert_hidden_dim)
    #uttrance_logits = uttrance_logits.permute(1,0,2)
    #_, uttrance_logits =  self.gru_utt(uttrance_logits)
    #uttrance_logits = uttrance_logits.view(-1, 10, self.hparams.bert_hidden_dim)
    
    #respones_logits = respones_logits.permute(1,0,2)
    #_,respones_logits = self.gru_res(respones_logits)
    
    #respones_logits = respones_logits.permute(1,0,2)
    #respones_logits = respones_logits.view(-1,1,self.hparams.bert_hidden_dim)
    
    utt_res_logits = torch.cat((uttrance_logits,respones_logits),1)
    utt_res_logits = utt_res_logits.permute(1,0,2)
    _, cls_logits = self.gru_all(utt_res_logits)
    
    cls_logits = cls_logits.view(-1, self.hparams.bert_hidden_dim)
    
    # response
    #response = respones_logits.view(-1,self.hparams.bert_hidden_dim,1) # version 2.0 b*d*1
    #att = torch.matmul(uttrance_logits, response)  # version 2.0.1   b*l*1
    #att  = self.softmax(att)  # version 2.0  b*l*1
    #uttrance_logits = torch.mul(uttrance_logits, att) # version 2.0.1
    #uttrance_logits = torch.matmul(att.permute(0,2,1), uttrance_logits)
    #uttrance_logits = uttrance_logits.view(-1,self.hparams.bert_hidden_dim)
    
    #cls_logits = torch.cat((uttrance_logits,respones_logits),1)
    
    #uttrance_logits = uttrance_logits.permute(1, 0, 2)
   
    #respones_logits = respones_logits.view(1, -1, self.hparams.bert_hidden_dim).contiguous()
    #_, cls_logits = self.gru(uttrance_logits, respones_logits)
    #cls_logits = cls_logits.view(-1, self.hparams.bert_hidden_dim)
    logits = self._classification(cls_logits) # bs, 1
    logits = logits.squeeze(-1)

    return logits
