import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hid_dim,
        n_layers,
        kernel_size,
        dropout,
        device,
        max_length=100,
    ):
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hid_dim,
                    out_channels=2 * hid_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                )

                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """Forward encoder pass
        Args:
            src (torch.tensor): [B , L] batch_size x source len input tensor

        B: Batch size
        L: Source length
        E: Embedding dim
        H: Hidden dim
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # pos = [0, 1, 2, 3, ..., src len - 1]
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )  # [B , L]

        tok_embedded = self.tok_embedding(src)  # [B , L , E]
        pos_embedded = self.pos_embedding(pos)  # [B , L , E]
        embedded = self.dropout(tok_embedded + pos_embedded)  # [B , L , E]

        conv_input = self.emb2hid(embedded)  # [B, L, H]
        conv_input = conv_input.permute(0, 2, 1)  # [B, H, L]

        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))  # [B, 2 * H, L]
            conved = F.glu(conved, dim=1)  # [B, H, L]
            conved = (conved + conv_input) * self.scale  # [B, H, L]
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))  # [B, L , E]
        combined = (conved + embedded) * self.scale  # [B, L, E]

        return conved, combined


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        hid_dim,
        n_layers,
        kernel_size,
        dropout,
        trg_pad_idx,
        device,
        max_length=100,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hid_dim,
                    out_channels=2 * hid_dim,
                    kernel_size=kernel_size,
                )

                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        """Attention layer
        Args:
            embedded (torch.tensor):         [B , TL , E] Embeded target tokens
            conved (torch.tensor):           [B , H , TL] Targets passed through Decoder CNN
            encoder_conved (torch.tensor):   [B , SL , E] Output of encoder conv block
            encoder_combined (torch.tensor): [B , SL , E] Output of encoder conv block + source embeddings

        B:  Batch size
        H:  Hidden dimension
        E:  Embedding dimension
        SL: Source Sequence Length
        TL: Target Sequence Length

        """
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))  # [B , TL , E]
        combined = (conved_emb + embedded) * self.scale  # [B , TL , E]

        energy = torch.matmul(
            combined, encoder_conved.permute(0, 2, 1)
        )  # [B , TL , SL]
        attention = F.softmax(energy, dim=2)  # [B , TL , SL]

        attended_encoding = torch.matmul(attention, encoder_combined)  # [B , TL , E]
        attended_encoding = self.attn_emb2hid(attended_encoding)  # [B , TL , H]

        attended_combined = (
            conved + attended_encoding.permute(0, 2, 1)
        ) * self.scale  # [B , H , TL]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        """Forward decoder pass
        Args:
            trg (torch.tensor): [B , TL] target sequence tokens
            encoder_conved (torch.tensor):   [B , SL , E] Output of encoder conv block
            encoder_combined (torch.tensor): [B , SL , E] Output of encoder conv block + source embeddings

        B:  Batch Size
        SL: Source sequence length
        TL: Target sequence length
        H:  Hidden dimension
        E:  Embedding dimension
        K:  Kernel size
        O:  Output dimension
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )  # [B , TL]

        tok_embedded = self.tok_embedding(trg)  # [B , TL , E]
        pos_embedded = self.pos_embedding(pos)  # [B , TL , E]
        embedded = self.dropout(tok_embedded + pos_embedded)  # [B , TL , E]

        conv_input = self.emb2hid(embedded)  # [B , TL , H]
        conv_input = conv_input.permute(0, 2, 1)  # [B , H , TL]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = (
                torch.zeros(batch_size, hid_dim, self.kernel_size - 1)
                .fill_(self.trg_pad_idx)
                .to(self.device)
            )

            padded_conv_input = torch.cat(
                (padding, conv_input), dim=2
            )  # [B , H , TL + K - 1]

            conved = conv(padded_conv_input)  # [B , 2*H , TL]
            conved = F.glu(conved, dim=1)  # [B , H , TL]

            attention, conved = self.calculate_attention(
                embedded, conved, encoder_conved, encoder_combined
            )  # [B , TL , SL]

            conved = (conved + conv_input) * self.scale  # [B , H , TL]
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))  # [B , TL , E]
        output = self.fc_out(self.dropout(conved))  # [B , TL , O]

        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]

        return output, attention
