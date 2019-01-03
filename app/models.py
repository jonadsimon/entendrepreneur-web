from app import db
import numpy as np


class Word(db.Model): # will likely cause a name collision... "GraphemePhonemePair"
    __tablename__ = 'words'

    id = db.Column(db.Integer, primary_key=True)
    grapheme = db.Column(db.String)
    phoneme = db.Column(db.ARRAY(db.String))
    grapheme_chunks = db.Column(db.JSON)
    phoneme_chunks = db.Column(db.JSON)

    def __repr__(self):
        return "<Word(grapheme='%s', phoneme='%s')>" % (self.grapheme, '-'.join(self.phoneme))

    def get_subgrapheme_from_subphoneme_inds(self, start_idx, end_idx, return_inds=False):
        '''
        Return the subgrapheme string (return_inds=False), or inclusive subgrapheme indices (return_inds=True),
        corresponding to the subphoneme starting at start_idx and ending at end_idx, inclusive
        '''
        chunk_lengths = map(len, self.phoneme_chunks)
        valid_end_inds = np.cumsum(chunk_lengths) - 1
        valid_start_inds = np.cumsum(chunk_lengths) - chunk_lengths

        if start_idx not in valid_start_inds:
            raise Exception('\'start_idx\' falls in the middle of a phoneme chunk')

        if end_idx not in valid_end_inds:
            raise Exception('\'end_idx\' falls in the middle of a phoneme chunk')

        # Include null-graphs at the boundaries
        start_chunk_idx = np.where(valid_start_inds == start_idx)[0].min()
        end_chunk_idx = np.where(valid_end_inds == end_idx)[0].max()

        if not return_inds:
            # Return the subgrapheme corresponding to the providing subphoneme indices
            subgrapheme = sum(map(list, self.grapheme_chunks[start_chunk_idx:end_chunk_idx+1]), [])
            return subgrapheme
        else:
            # Return the subgrapheme *indices* corresponding to the providing subphoneme indices
            subgrapheme_start_idx = sum(map(len, self.grapheme_chunks[:start_chunk_idx]))
            subgrapheme_end_idx = sum(map(len, self.grapheme_chunks[:end_chunk_idx+1])) - 1
            return subgrapheme_start_idx, subgrapheme_end_idx

    def get_destressed_phoneme(self):
        '''
        Strip the stress information off of the phoneme
        '''
        return [filter(str.isalpha, str(phone)) for phone in self.phoneme]


class SubgraphemeFrequency(db.Model):
    __tablename__ = 'subgrapheme_frequencies'

    id = db.Column(db.Integer, primary_key=True)
    grapheme = db.Column(db.Text, index=True, unique=True) # needs to be db.Text rather than db.String (i.e. varchar) to avoid typing issues
    frequency = db.Column(db.Integer, default=1)
    frequency_head = db.Column(db.Integer, default=1)
    frequency_tail = db.Column(db.Integer, default=1)

    def __repr__(self):
        return "<SubgraphemeFrequency(grapheme='%s', frequency=%i, frequency_head=%i, frequency_tail=%i)>" % (self.grapheme, self.frequency, self.frequency_head, self.frequency_tail)

    @classmethod
    def get_subgrapheme_frequency(cls, this_grapheme, side='all'):
        '''
        Return the frequency of the grapheme
        '''
        try:
            subgrapheme_frequency = cls.query.filter_by(grapheme=this_grapheme).one()
        except:
            # if the grapheme is not present in the database i.e. because it is too long, return the default frequency of 1
            return 1
        if side == 'head':
            return subgrapheme_frequency.frequency_head
        elif side == 'tail':
            return subgrapheme_frequency.frequency_tail
        elif side == 'all':
            return subgrapheme_frequency.frequency
        else:
            raise "Argument 'side' must be either 'head', 'tail', or 'all'"


class SubphonemeFrequency(db.Model):
    __tablename__ = 'subphoneme_frequencies'

    id = db.Column(db.Integer, primary_key=True)
    phoneme = db.Column(db.ARRAY(db.Text), index=True, unique=True) # needs to be db.Text rather than db.String (i.e. varchar) to avoid typing issues
    frequency = db.Column(db.Integer, default=1)
    frequency_head = db.Column(db.Integer, default=1)
    frequency_tail = db.Column(db.Integer, default=1)

    def __repr__(self):
        return "<SubphonemeFrequency(phoneme='%s', frequency=%i, frequency_head=%i, frequency_tail=%i)>" % (self.phoneme, self.frequency, self.frequency_head, self.frequency_tail)

    @classmethod
    def get_subphoneme_frequency(cls, this_phoneme, side='all'):
        '''
        Return the frequency of the phoneme
        '''
        try:
            subphoneme_frequency = cls.query.filter_by(phoneme=this_phoneme).one()
        except:
            # if the phoneme is not present in the database i.e. because it is too long, return the default frequency of 1
            return 1
        if side == 'head':
            return subphoneme_frequency.frequency_head
        elif side == 'tail':
            return subphoneme_frequency.frequency_tail
        elif side == 'all':
            return subphoneme_frequency.frequency
        else:
            raise "Argument 'side' must be either 'head', 'tail', or 'all'"


class FasttextVector(db.Model):
    __tablename__ = 'fasttext_vectors'

    id = db.Column(db.Integer, primary_key=True)
    grapheme = db.Column(db.String, index=True, unique=True)
    v1 = db.Column(db.Float)
    v2 = db.Column(db.Float)
    v3 = db.Column(db.Float)
    v4 = db.Column(db.Float)
    v5 = db.Column(db.Float)
    v6 = db.Column(db.Float)
    v7 = db.Column(db.Float)
    v8 = db.Column(db.Float)
    v9 = db.Column(db.Float)
    v10 = db.Column(db.Float)
    v11 = db.Column(db.Float)
    v12 = db.Column(db.Float)
    v13 = db.Column(db.Float)
    v14 = db.Column(db.Float)
    v15 = db.Column(db.Float)
    v16 = db.Column(db.Float)
    v17 = db.Column(db.Float)
    v18 = db.Column(db.Float)
    v19 = db.Column(db.Float)
    v20 = db.Column(db.Float)
    v21 = db.Column(db.Float)
    v22 = db.Column(db.Float)
    v23 = db.Column(db.Float)
    v24 = db.Column(db.Float)
    v25 = db.Column(db.Float)
    v26 = db.Column(db.Float)
    v27 = db.Column(db.Float)
    v28 = db.Column(db.Float)
    v29 = db.Column(db.Float)
    v30 = db.Column(db.Float)
    v31 = db.Column(db.Float)
    v32 = db.Column(db.Float)
    v33 = db.Column(db.Float)
    v34 = db.Column(db.Float)
    v35 = db.Column(db.Float)
    v36 = db.Column(db.Float)
    v37 = db.Column(db.Float)
    v38 = db.Column(db.Float)
    v39 = db.Column(db.Float)
    v40 = db.Column(db.Float)
    v41 = db.Column(db.Float)
    v42 = db.Column(db.Float)
    v43 = db.Column(db.Float)
    v44 = db.Column(db.Float)
    v45 = db.Column(db.Float)
    v46 = db.Column(db.Float)
    v47 = db.Column(db.Float)
    v48 = db.Column(db.Float)
    v49 = db.Column(db.Float)
    v50 = db.Column(db.Float)
    v51 = db.Column(db.Float)
    v52 = db.Column(db.Float)
    v53 = db.Column(db.Float)
    v54 = db.Column(db.Float)
    v55 = db.Column(db.Float)
    v56 = db.Column(db.Float)
    v57 = db.Column(db.Float)
    v58 = db.Column(db.Float)
    v59 = db.Column(db.Float)
    v60 = db.Column(db.Float)
    v61 = db.Column(db.Float)
    v62 = db.Column(db.Float)
    v63 = db.Column(db.Float)
    v64 = db.Column(db.Float)
    v65 = db.Column(db.Float)
    v66 = db.Column(db.Float)
    v67 = db.Column(db.Float)
    v68 = db.Column(db.Float)
    v69 = db.Column(db.Float)
    v70 = db.Column(db.Float)
    v71 = db.Column(db.Float)
    v72 = db.Column(db.Float)
    v73 = db.Column(db.Float)
    v74 = db.Column(db.Float)
    v75 = db.Column(db.Float)
    v76 = db.Column(db.Float)
    v77 = db.Column(db.Float)
    v78 = db.Column(db.Float)
    v79 = db.Column(db.Float)
    v80 = db.Column(db.Float)
    v81 = db.Column(db.Float)
    v82 = db.Column(db.Float)
    v83 = db.Column(db.Float)
    v84 = db.Column(db.Float)
    v85 = db.Column(db.Float)
    v86 = db.Column(db.Float)
    v87 = db.Column(db.Float)
    v88 = db.Column(db.Float)
    v89 = db.Column(db.Float)
    v90 = db.Column(db.Float)
    v91 = db.Column(db.Float)
    v92 = db.Column(db.Float)
    v93 = db.Column(db.Float)
    v94 = db.Column(db.Float)
    v95 = db.Column(db.Float)
    v96 = db.Column(db.Float)
    v97 = db.Column(db.Float)
    v98 = db.Column(db.Float)
    v99 = db.Column(db.Float)
    v100 = db.Column(db.Float)
    v101 = db.Column(db.Float)
    v102 = db.Column(db.Float)
    v103 = db.Column(db.Float)
    v104 = db.Column(db.Float)
    v105 = db.Column(db.Float)
    v106 = db.Column(db.Float)
    v107 = db.Column(db.Float)
    v108 = db.Column(db.Float)
    v109 = db.Column(db.Float)
    v110 = db.Column(db.Float)
    v111 = db.Column(db.Float)
    v112 = db.Column(db.Float)
    v113 = db.Column(db.Float)
    v114 = db.Column(db.Float)
    v115 = db.Column(db.Float)
    v116 = db.Column(db.Float)
    v117 = db.Column(db.Float)
    v118 = db.Column(db.Float)
    v119 = db.Column(db.Float)
    v120 = db.Column(db.Float)
    v121 = db.Column(db.Float)
    v122 = db.Column(db.Float)
    v123 = db.Column(db.Float)
    v124 = db.Column(db.Float)
    v125 = db.Column(db.Float)
    v126 = db.Column(db.Float)
    v127 = db.Column(db.Float)
    v128 = db.Column(db.Float)
    v129 = db.Column(db.Float)
    v130 = db.Column(db.Float)
    v131 = db.Column(db.Float)
    v132 = db.Column(db.Float)
    v133 = db.Column(db.Float)
    v134 = db.Column(db.Float)
    v135 = db.Column(db.Float)
    v136 = db.Column(db.Float)
    v137 = db.Column(db.Float)
    v138 = db.Column(db.Float)
    v139 = db.Column(db.Float)
    v140 = db.Column(db.Float)
    v141 = db.Column(db.Float)
    v142 = db.Column(db.Float)
    v143 = db.Column(db.Float)
    v144 = db.Column(db.Float)
    v145 = db.Column(db.Float)
    v146 = db.Column(db.Float)
    v147 = db.Column(db.Float)
    v148 = db.Column(db.Float)
    v149 = db.Column(db.Float)
    v150 = db.Column(db.Float)

    def __repr__(self):
        return "<FasttextVectorElement(grapheme_id=%i, v1=%d, v2=%d, ... , v150=%d)>" % (self.grapheme_id, self.v1, self.v2, self.v150)
