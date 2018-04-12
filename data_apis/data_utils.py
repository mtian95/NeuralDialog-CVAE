#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import numpy as np


# Data feed
class LongDataLoader(object):
    """A special efficient data loader for TBPTT"""
    batch_size = 0
    backward_size = 0
    step_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    grid_indexes = None
    indexes = None
    data_lens = None
    data_size = None
    prev_alive_size = 0
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, backward_size, step_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0 # pointer
        self.batch_size = batch_size # currently set to 30 in config
        self.backward_size = backward_size
        self.step_size = step_size
        self.prev_alive_size = batch_size

        # create batch indexes
        print "data size", self.data_size
        print "batch size", self.batch_size

        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        # batch_indexes = list of lists of the indices in each batch. one list for each batch. 
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])



        left_over = self.data_size-temp_num_batch*batch_size

        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []
        # idx is the batch number, b_ids is a list of all the data indices in this batch
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids] # lengths of the dialogs in this batch 
            max_len = self.data_lens[b_ids[-1]] 
            min_len = self.data_lens[b_ids[0]]
            assert np.max(all_lens) == max_len # max_len = max number of sentences in a dialog in this batch
            assert np.min(all_lens) == min_len

            # segment start and end indicies. Each segment should be of len backward_size if possible (sometimes shorter)
            num_seg = (max_len-self.backward_size) // self.step_size 
            # print("number of segments " + str(num_seg))
            if num_seg > 0:
                cut_start = range(0, num_seg*self.step_size, step_size)
                cut_end = range(self.backward_size, num_seg*self.step_size+self.backward_size, step_size)
                assert cut_end[-1] < max_len
                cut_start = [0] * (self.backward_size-2) +cut_start # since we give up on the seq training idea
                cut_end = range(2, self.backward_size) + cut_end
            else:
                cut_start = [0] * (max_len-2)
                cut_end = range(2, max_len)

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len-1]
            if intra_shuffle and shuffle:
               np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)

        self.num_batch = len(self.grid_indexes) #TODO fix this. make 1. when just trying out
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
        else:
            return None


class SWDADataLoader(LongDataLoader):
    # TODO change this none
    def __init__(self, name, data, meta_data, config, lda_model=None):
        assert len(data) == len(meta_data)
        self.name = name
        self.data = data # from api.get_dialog_corpus()
        self.meta_data = meta_data # from api.get_meta_corpus()
        self.data_size = len(data)
        self.lda_model = lda_model
        # length of each dialog aka num of sentences
        self.data_lens = all_lens = [len(line) for line in self.data]
        self.max_utt_size = config.max_utt_len
        print("Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                            np.min(all_lens),
                                                            float(np.mean(all_lens))))
        # self.indexes gets a list of indices that would sort the dialogs in increasing order of sent number
        self.indexes = list(np.argsort(all_lens))

    # pads tokens to be max_utt_size with the "pad" character in corpus
    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size-len(tokens))
        else:
            return tokens

    """
        Extracts all the info for a grid (aka a segment) in a batch.
        A grid is a batch of data that only looks at a segment of the dialogs in each batch. 
        b_id = index of the batch being prepared
        s_id = starting point of segment within a dialog (ex: take the first 5 sentences in dialog to predict the 6th)
        e_id = ending point of segment within a dialog 

        batch_indexes = list of lists of the data indices in each batch
        batch_ids = list of all data indices included in batch b_id
        rows = list of dialogs in this batch. from api.get_dialog_corpus()
            Each dialog is a list of sentences. 
            Each sentence = (sent in tokens, 0/1 paragraph end label, vector version of topic of just that sent)
        meta_rows = list of meta info for the data in this batch. from api.get_meta_corpus()
            Each paragraph has its own meta data = topic of that entire paragraph
        dialog_lens = list of number of sentences in each dialog in this batch

        out_row = sentence being predicted using in_row as context
        floors = list of lists of paragraph end labels (1 if sent is last in par, else 0)
        out_das = sentence topic of the sentence being predicted (out_row)
    """
    def _prepare_batch(self, cur_grid, prev_grid):
        b_id, s_id, e_id = cur_grid
        print "starting id", s_id

        batch_ids = self.batch_indexes[b_id]
        rows = [self.data[idx] for idx in batch_ids]
        meta_rows = [self.meta_data[idx] for idx in batch_ids]
        dialog_lens = [self.data_lens[idx] for idx in batch_ids]

        # topics = np.array([meta[2] for meta in meta_rows])
        cur_pos = [np.minimum(1.0, e_id/float(l)) for l in dialog_lens]

        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, floors, out_utts, out_lens, out_floors, out_das = [], [], [], [], [], [], []
        for row in rows:
            if s_id < len(row)-1:
                cut_row = row[s_id:e_id]

                # split rows into input (in_row) and the thing being predicted (out_row)
                in_row = cut_row[0:-1]
                out_row = cut_row[-1]
                out_utt, out_floor, out_sent_topic = out_row

                context_utts.append([self.pad_to(utt) for utt, floor, sent_topic in in_row])
                floors.append([floor for utt, floor, sent_topic in in_row])
                # floors.append([int(floor==out_floor) for utt, floor, sent_topic in in_row])
                context_lens.append(len(cut_row) - 1)

                out_utt = self.pad_to(out_utt, do_pad=False)
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))
                out_floors.append(out_floor)
                out_das.append(out_sent_topic)

            else:
                print(row)
                raise ValueError("S_ID %d larger than row" % s_id)


        # my_profiles = np.array([meta[out_floors[idx]] for idx, meta in enumerate(meta_rows)])
        # ot_profiles = np.array([meta[1-out_floors[idx]] for idx, meta in enumerate(meta_rows)])
        vec_paragraph_topics = np.array(meta_rows)
        vec_context_lens = np.array(context_lens)
        vec_out_floors = np.array(out_floors).reshape(len(out_floors), 1)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens), self.max_utt_size), dtype=np.int32)
        vec_floors = np.zeros((self.batch_size, np.max(vec_context_lens)), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_out_das = np.array(out_das)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_floors[b_id, 0:vec_context_lens[b_id]] = floors[b_id]
            vec_context[b_id, 0:vec_context_lens[b_id], :] = np.array(context_utts[b_id])

        # return vec_context, vec_context_lens, vec_floors, topics, my_profiles, ot_profiles, vec_outs, vec_out_lens, vec_out_das
        return vec_context, vec_context_lens, vec_floors, vec_outs, vec_out_lens, vec_out_das, vec_paragraph_topics, vec_out_floors









