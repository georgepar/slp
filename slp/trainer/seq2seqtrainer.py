import torch.nn as nn
import os
import torch
import torch.nn.functional as F

def train(train_batches, model, model_optimizer, criterion, clip=None):
    """
    This function is used to train a Seq2Seq model.
    Model optimizer can be a list of optimizers if wanted(e.g. if we want to
    have different lr for encoder and decoder).
    """

    if not isinstance(model_optimizer, list):
        model_optimizer.zero_grad()
    else:
        for optimizer in model_optimizer:
            optimizer.zero_grad()
    epoch_loss = 0
    for index, batch in enumerate(train_batches):

        inputs, lengths_inputs, targets, masks_targets = batch
        inputs = inputs.long().cuda()
        targets = targets.long().cuda()
        lengths_inputs.cuda()
        masks_targets.cuda()

        if not isinstance(model_optimizer, list):
            model_optimizer.zero_grad()
        else:
            for optimizer in model_optimizer:
                optimizer.zero_grad()

        decoder_outputs = model(inputs, lengths_inputs, targets)

        # calculate and accumulate loss
        # loss = 0
        # n_totals = 0
        # for time in range(0, len(decoder_outputs)):
        #
        #     loss += criterion(decoder_outputs[time], targets[:, time].long())
        #     n_totals += 1
        # loss.backward()
        #
        # epoch_loss += loss.item() / n_totals

        loss = criterion(decoder_outputs,targets)
        epoch_loss += loss.item()
        loss.backward()
        # Clip gradients: gradients are modified in place
        if clip is not None:
            _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Adjust model weights
        if not isinstance(model_optimizer, list):
            model_optimizer.step()
        else:
            for optimizer in model_optimizer:
                optimizer.step()

        last = index
    # we return average epoch loss
    return epoch_loss/(last+1)


def train_epochs(training_batches, model, model_optimizer,
                 criterion, num_epochs, print_every=1,
                 clip=None):

    print("Training...")
    model.train()

    # directory = os.path.join(save_dir, model_name)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # infofile = open(os.path.join(directory, 'model_info.txt'), 'w')
    # print(model_name, file=infofile)
    # print("Model architecture:  ", model,file=infofile)
    # print("Model optimizer:     ", model_optimizer,file=infofile)
    # print("Loss function:       ",criterion,file=infofile)
    # infofile.close()

    # logfile = open(os.path.join(directory, 'log_file.txt'), 'w')
    for epoch in range(num_epochs+1):
        avg_epoch_loss = 0

        # Train to all batches
        if clip is not None:
            avg_epoch_loss = train(training_batches, model, model_optimizer,
                                   criterion, clip)
        else:
            avg_epoch_loss = train(training_batches, model, model_optimizer,
                                   criterion)

        # Print progress
        if epoch % print_every == 0:
            print("Epoch {}; Percent complete: {:.1f}%; Average Epoch loss: {"
                  ":.4f}".format(epoch, epoch / num_epochs * 100,
                                 avg_epoch_loss))
                  #file=logfile)

    #     if save_every is not None:
    #         # Save checkpoint
    #         if epoch % save_every == 0:
    #
    #             if isinstance(model_optimizer, list):
    #                 torch.save({
    #                     'epoch': epoch,
    #                     'model': model.state_dict(),
    #                     'model_opt_enc': model_optimizer[0].state_dict(),
    #                     'model_opt_dec': model_optimizer[1].state_dict(),
    #                     'loss': avg_epoch_loss,
    #
    #                 }, os.path.join(directory, '{}_{}.tar'.format(epoch,
    #                                                               'checkpoint')))
    #             else:
    #                 torch.save({
    #                     'epoch': epoch,
    #                     'model': model.state_dict(),
    #                     'model_opt': model_optimizer.state_dict(),
    #                     'loss': avg_epoch_loss,
    #
    #                 }, os.path.join(directory, '{}_{}.tar'.format(epoch,
    #                                                               'checkpoint')))
    # logfile.close()


def validate( val_batches, model):
    """
    This function is used for validating the model!
    Model does not use "forward" but "evaluate , because we don't want to use
    teacher forcing!
    :param val_batches: batches given for validation
    :param model: trained model that need to have evaluate function (like
    forward)
    :return:
    """

    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(val_batches):
            inputs, lengths_inputs, targets, masks_targets = batch
            inputs = inputs.long().cuda()
            targets = targets.long().cuda()
            lengths_inputs.cuda()
            masks_targets.cuda()

            decoder_outputs, decoder_hidden = model.evaluate(inputs,
                                                             lengths_inputs)
            # decoder_outputs is a 3d tensor(batchsize,seq length,outputsize)

            for batch_index in range(decoder_outputs.shape[0]):
                out_logits = F.log_softmax(decoder_outputs[batch_index],dim=1)
                _,out_indexes = torch.max(out_logits,dim=1)

                print("Question: ", inputs[batch_index])
                print("Target: ",targets[batch_index])
                print("Response: ",out_indexes)

                print("+++++++++++++++++++++")


def inputInteraction(model, vocloader, text_preprocessor, text_tokenizer,
                     idx_loader, padder):
    max_len = model.max_target_len
    input_sentence = ""
    while True:
        try:
            # Get input response:
            input_sentence = input('>')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Process sentence
            input_filt = text_preprocessor.process_text([input_sentence])
            input_tokens = text_tokenizer.word_tokenization(input_filt)

            input_indexes = idx_loader.get_indexes(input_tokens)

            input_length = [len(input_indexes[0])]
            padded_input = padder.zeroPadding(input_indexes,max_len)
            padded_input = torch.LongTensor(padded_input).cuda()

            input_length = torch.LongTensor(input_length).cuda()

            dec_outs,dec_hidden = model.evaluate(padded_input,input_length)

            out_logits = F.log_softmax(dec_outs[0], dim=1)
            _, out_indexes = torch.max(out_logits, dim=1)
            #print(out_indexes)
            decoded_words = [vocloader.idx2word[int(index)] for index in
                             out_indexes]
            print("Response: ", decoded_words)

            print("+++++++++++++++++++++")

        except KeyError:
            print("Error:Encountered unknown word")