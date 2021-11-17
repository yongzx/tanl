import torch
from torch import  nn
from transformers import AutoModelForSeq2SeqLM, AutoConfig


class PrefixTuningBart(nn.Module):
    def __init__(self, model_name, cache_dir, prefix_seqlen=5, prefix_mid_dim=512, prefix_dropout=0.0):
        print('under the PrefixTuning model')
        super(PrefixTuningBart, self).__init__()
        assert "bart" in model_name
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.use_prefix = True
        self.config.preseqlen = self.prefix_seqlen = prefix_seqlen
        self.prefix_mid_dim = prefix_mid_dim
        self.prefix_dropout = prefix_dropout
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, config=self.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.match_n_layer = self.config.decoder_layers
        self.match_n_head = self.config.decoder_attention_heads
        self.n_embd = self.config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head  # embed_size_per_head

        
        # print('UNDER PARAMETRIZATION 1')

        self.input_tokens = torch.arange(self.prefix_seqlen).long()
        self.wte = nn.Embedding(self.prefix_seqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.prefix_mid_dim),
            nn.Tanh(),
            nn.Linear(self.prefix_mid_dim, self.match_n_layer * 2 * self.n_embd))
        
        self.use_encoder_prefix = True
        self.use_cross_prefix = True

        if self.use_encoder_prefix:
            self.wte_enc = nn.Embedding(self.prefix_seqlen, self.n_embd)
            self.control_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.prefix_mid_dim),
                nn.Tanh(),
                nn.Linear(self.prefix_mid_dim, self.match_n_layer * 2 * self.n_embd))

        if self.use_cross_prefix:
            self.wte2 = nn.Embedding(self.prefix_seqlen, self.n_embd)
            self.control_trans2 = nn.Sequential(
                nn.Linear(self.n_embd, self.prefix_mid_dim),
                nn.Tanh(),
                nn.Linear(self.prefix_mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.dropout = nn.Dropout(self.prefix_dropout)


    def get_prompt(self, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        print(past_key_values.shape)  # [batchsize, sequence length, decoder_layers * 2, decoder_attention_heads,
                                      # d_model // decoder_attention_heads]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        print(len(past_key_values), past_key_values[0].shape)   
            # len of decoder_layers, 
            # [2, batchsize, decoder_attention_heads, sequence length, d_model // decoder_attention_heads]


        if self.use_cross_prefix:  # xsum = True
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)


        if self.use_encoder_prefix: # xsum = True
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, prefix_seqlen
                                 },
                        }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                        }
            result.append(temp_dict)

        return result
    
    def forward(self,
        input_ids=None,
        past_key_values=None,
        **kwargs,
        ):
        past_key_values_prompt = self.get_prompt(bsz=input_ids.shape[0])  # xsum = get_prompt_p5

        # print(past_key_values)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt
        
        past_key_values = past_key_values_prompt

        output = self.base_model(input_ids=input_ids, 
                                 past_key_values=past_key_values,
                                 **kwargs)

        return output


class PrefixTuning(nn.Module):
    # TODO: need to rewrite the underlying transformers code

    def __init__(self, config):
        print('under the PrefixTuning model')
        super(PrefixTuning, self).__init__()
        # self.base_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.match_n_layer = 12
        self.match_n_head = 12
        self.n_embd = 768
        self.match_n_embd = self.n_embd // self.match_n_head  # embed_size_per_head

        self.preseqlen = 5
        self.mid_dim = 512
        self.prefix_dropout = 0.0
        # print('UNDER PARAMETRIZATION 1')

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
        
        self.use_encoder_prefix = False
        self.use_cross_prefix = True

        if self.use_encoder_prefix:
            self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
            self.control_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        if self.use_cross_prefix:
            self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
            self.control_trans2 = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.dropout = nn.Dropout(self.prefix_dropout)

    def get_prompt(self, bsz=None, sample_size=1):

        # note : for T5ForConditionalGeneration 
        # past_key_values (tuple(tuple(torch.FloatTensor)) of length config.n_layers 
        # with each tuple having 4 tensors of shape (batch_size, num_heads, sequence_length - 1, embed_size_per_head)) –
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        print("self.input_tokens", self.input_tokens.shape)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(self.match_n_layer, 2, bsz, self.match_n_head, seqlen, self.match_n_embd)
        print(past_key_values.shape)
        # should have [config.n_layers, 2, batchsize, decoder_attention_heads, seqlen, d_model // decoder_attention_heads]
        

        if self.use_cross_prefix:  # xsum = True
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(self.match_n_layer, 2, bsz, self.match_n_head, seqlen, self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            print("cross.attention:", past_key_values2.shape)   

            result = torch.cat((past_key_values, past_key_values2), 1)
            print("final:", len(result), result[0].shape)   
            return result[:, :, :, :, :-1, :]

        # if self.use_encoder_prefix: # xsum = True
        #     input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
        #     temp_control_enc = self.wte_enc(input_tokens_enc)
        #     past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        #     bsz_enc, seqlen, _ = past_key_values_enc.shape
        #     past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
        #                                              self.match_n_embd)
        #     past_key_values_enc = self.dropout(past_key_values_enc)
        #     past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                 },
                        }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                        }
            result.append(temp_dict)

        return (result if result else None)
    
    def forward(self,
        input_ids=None,
        past_key_values=None,
        **kwargs,
        ):
        print("kwargs:", kwargs)

        print("self.get_prompt =======")
        past_key_values_prompt = self.get_prompt(bsz=input_ids.shape[0])  # xsum = get_prompt_p5

        # print(past_key_values)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt
        
        past_key_values = past_key_values_prompt

        output = self.base_model(input_ids=input_ids, 
                                 past_key_values=past_key_values,
                                 **kwargs)

        return output