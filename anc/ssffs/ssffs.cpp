/*=========================================================================
  ssffs.cpp — standalone Stochastic Sequential Forward Floating Search (sSFFS)

  Accompanying material for the paper
    "Stochastic Sequential Search in Very-High-Dimensional Feature Selection"
    (P. Somol, J. Grim).

  This is a SELF-CONTAINED reimplementation of the sSFFS experiment stack:
  it depends on nothing but a C++17 compiler and reproduces, bit for bit,
  the results of the reference implementation in the Feature Selection
  Toolbox 4 (FST4, http://fst.utia.cz) for the two criterion configurations
  used in the paper's experiments:

    (a) wrapper: k-NN classification accuracy estimated by k-fold
        cross-validation on the training part of a random train/test split;
    (b) filter: multinomial Bhattacharyya distance estimated on the
        training part of a random train/test split.

  The option surface is deliberately reduced to what the paper's experiments
  need. Data format: ARFF only (dense or sparse rows; numeric features;
  nominal class attribute named 'class'). Randomness: an own portable PRNG
  (the Windows-UCRT rand() recurrence), so a given --seed pins the data
  split and the whole search trajectory on every platform.

  Build (use the exact flags for bit-reproducibility of floating point):
      c++ -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG -o ssffs ssffs.cpp

  Example (the paper's wrapper configuration, bounded run):
      ./ssffs --data madelon_train_nips_500.arff \
              --rr-train 50 --rr-test 50 --cv-folds 3 --scaler to01 \
              --criterion wrapper-knn --knn-k 1 \
              --target-d 20 --sffs-delta 5 \
              --step-cap 100 --step-cap-backward 50 \
              --warmup-probes 200 --warmup-card 10 --seed 1

  Example (the paper's filter configuration, bounded run):
      ./ssffs --data reuters_apte.arff \
              --rr-train 50 --rr-test 40 --scaler void \
              --criterion multinom-bhattacharyya \
              --target-d 25 --sffs-delta 5 \
              --step-cap 100 --step-cap-backward 50 \
              --warmup-probes 2000 --warmup-card 25 --seed 1

  Output: one JSON line per improved solution on stderr
  ({"event":"solution",...}, identical to fst4 --progress json), a final
  JSON result on stdout ({"value","size","features","evaluations"}).

  Copyright (c) 2026 Institute of Information Theory and Automation,
  The Czech Academy of Sciences (UTIA)

  Licensed under the MIT License (the full text is reproduced below and in
  the accompanying LICENSE file, so this single file remains self-contained):

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=========================================================================*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <sstream>

// ------------------------------------------------------------------ PRNG
// The Windows-UCRT rand() recurrence (LCG), 15-bit output. Identical to
// FST4's fst_rand(): after seed 1 the first five draws are
// 41, 18467, 6334, 26500, 19169.
static unsigned int g_rng_state = 1u;
static inline void ss_srand(unsigned int seed) { g_rng_state = seed; }
static inline int ss_rand() {
	g_rng_state = g_rng_state * 214013u + 2531011u;
	return (int)((g_rng_state >> 16) & 0x7fff);
}
static const int SS_RAND_MAX = 32767;
static inline double frand() { return (double)ss_rand() / ((double)SS_RAND_MAX + 1.0); } // uniform [0,1)

// ------------------------------------------------------------------ ARFF
// Minimal ARFF reader covering the dialect the paper's data files use:
// numeric attributes (stored as float, exactly as the reference reader
// does — values are widened to double only when copied into the working
// array), one nominal attribute named 'class' (case-insensitive, quotes
// tolerated) anywhere in the attribute list, dense or sparse data rows,
// '%' comments and blank lines. Anything else is a fatal error.
struct ArffAttribute {
	std::string name;
	bool nominal = false;
	bool is_class = false;
	std::vector<std::string> values; // nominal value list
};

struct ArffData {
	unsigned int n_features = 0;             // non-class attributes
	unsigned int n_classes = 0;
	std::vector<unsigned int> class_size;    // per class
	std::vector<double> data;                // class-major [class][sample][feature]
};

static char *arff_trim(char *s) {
	char *e = s + strlen(s) - 1;
	while (e >= s && (*e == ' ' || *e == 9 || *e == 13 || *e == 10)) *e-- = 0;
	while (*s == ' ' || *s == 9 || *s == 13 || *s == 10) s++;
	return s;
}

static bool arff_read_line(FILE *f, std::vector<char> &buff) {
	size_t len = 0; buff[0] = 0;
	while (fgets(buff.data() + len, (int)(buff.size() - len), f) != nullptr) {
		len += strlen(buff.data() + len);
		if ((len > 0 && buff[len - 1] == '\n') || feof(f)) return true;
		if (len >= buff.size() - 2) buff.resize(2 * buff.size());
	}
	return len > 0;
}

static std::string upper(const std::string &s) {
	std::string r(s);
	for (char &c : r) c = (char)toupper((unsigned char)c);
	return r;
}

[[noreturn]] static void die(const std::string &msg) {
	std::cerr << "ssffs: " << msg << std::endl;
	std::exit(1);
}

static ArffData load_arff(const std::string &path) {
	FILE *f = fopen(path.c_str(), "rt");
	if (!f) die("cannot open '" + path + "'");
	std::vector<char> buff(8192);

	std::vector<ArffAttribute> attrs;
	int class_attr = -1;
	bool reading_data = false;

	// one record = all loaded attribute values (features as float, class id)
	struct Record { std::vector<float> feat; int cls = -1; };
	std::vector<Record> records;
	std::vector<int> feat_of_attr;   // attribute index -> feature position (-1 for class)
	unsigned int n_features = 0, n_classes = 0;

	auto prepare_attributes = [&]() {
		for (size_t i = 0; i < attrs.size(); i++) {
			const std::string un = upper(attrs[i].name);
			if ((un == "CLASS" || un == "'CLASS'") && attrs[i].nominal) {
				attrs[i].is_class = true;
				class_attr = (int)i;      // last match wins, as in the reference
			}
		}
		feat_of_attr.assign(attrs.size(), -1);
		for (size_t i = 0; i < attrs.size(); i++)
			if (!attrs[i].is_class) feat_of_attr[i] = (int)n_features++;
		if (class_attr < 0) die("no nominal 'class' attribute found");
		n_classes = (unsigned int)attrs[(size_t)class_attr].values.size();
	};

	// parse one attribute's value token into the record (mirrors readAttrValue)
	auto read_attr_value = [&](size_t ai, Record &rec, char *value) {
		ArffAttribute &a = attrs[ai];
		if (!a.nominal) {
			float v;
			if (sscanf(value, "%f", &v) != 1) die("bad numeric value '" + std::string(value) + "' for attribute " + a.name);
			rec.feat[(size_t)feat_of_attr[ai]] = v;
		} else {
			int val = -1;
			for (size_t j = 0; j < a.values.size(); j++)
				if (a.values[j] == value) { val = (int)j; break; }
			if (val == -1) die("nominal value '" + std::string(value) + "' not found for attribute " + a.name);
			if (a.is_class) rec.cls = val;
			else die("nominal feature attributes are not supported ('" + a.name + "')");
		}
	};

	while (arff_read_line(f, buff)) {
		char *line = arff_trim(buff.data());
		if (!line[0]) continue;
		if (line[0] == '%') continue;
		if (reading_data) {
			line = arff_trim(line);
			if (!*line) continue;
			Record rec;
			rec.feat.assign(n_features, 0.0f);
			if (*line == '{') { // sparse row (mirrors readSparseRow)
				char *row = line + 1;
				bool ok = true;
				while (*row != 0 && *row != '}' && ok) {
					while (*row == 9 || *row == ' ') row++;
					char *id = row;
					while (*row != 9 && *row != ' ' && *row != 0) row++;
					if (*row == 0) { ok = false; break; }
					*row = 0; row++;
					while (*row == 9 || *row == ' ') row++;
					char *val = row;
					if (*row == '"' || *row == '\'') {
						row++;
						while (*row != 0 && *row != *val) row++;
						if (*row == 0) { ok = false; break; }
						*row = 0; val++; row++;
						while (*row == 9 || *row == ' ' || *row == ',') row++;
					} else {
						while (*row != ',' && *row != '}' && *row != 0) row++;
						if (*row == 0) { ok = false; break; }
						if (*row == '}') { *row = 0; row++; *row = '}'; }
						else { *row = 0; row++; }
						val = arff_trim(val);
					}
					int att_id = atoi(id);
					if (att_id < 0 || att_id >= (int)attrs.size()) die("sparse row: invalid attribute index");
					read_attr_value((size_t)att_id, rec, val);
					while (*row == 9 || *row == ' ') row++;
				}
				if (!ok) die("malformed sparse data row");
				if (rec.cls == -1) rec.cls = 0; // omitted class attribute = class 0
			} else {            // dense row (mirrors readNormalRow)
				char *row = line;
				for (size_t ci = 0; ci < attrs.size(); ci++) {
					if (*row == 0) die("dense row ended too soon");
					char *p = row;
					char *q = row; while (*q == ' ' || *q == 9) q++;
					if (*q == '\'' || *q == '"') { const char qc = *q; q++; while (*q != 0 && *q != qc) q++; if (*q) q++; row = q; }
					while (*row != 0 && *row != ',') row++;
					char tmpch = *row; *row = 0;
					read_attr_value(ci, rec, arff_trim(p));
					*row = tmpch;
					if (*row == ',') row++;
				}
				if (rec.cls < 0) die("dense row without class value");
			}
			records.push_back(std::move(rec));
		} else if (line[0] == '@') {
			char *val = line + 2; while (*val != ' ' && *val != 0 && *val != 9) val++; *val = 0; val++;
			std::string key = upper(line);
			if (key == "@RELATION") continue;
			if (key == "@DATA") { reading_data = true; prepare_attributes(); continue; }
			if (key == "@ATTRIBUTE") {
				char *info = arff_trim(val);
				if (!*info) die("empty attribute definition");
				char *tv = info + 1; while (*tv != ' ' && *tv != 0 && *tv != 9) tv++; *tv = 0; tv++;
				tv = arff_trim(tv);
				ArffAttribute a;
				a.name = info;
				if (*tv == '{') {
					a.nominal = true;
					tv++;
					char *p = tv;
					while (*tv != '}' && *tv != 0) {
						if (*tv == ',') {
							*tv = 0;
							std::vector<char> t(p, p + strlen(p) + 1);
							a.values.push_back(std::string(arff_trim(t.data())));
							p = ++tv;
						}
						tv++;
					}
					if (*tv == 0) die("malformed nominal attribute '" + a.name + "'");
					*tv = 0;
					std::vector<char> t(p, p + strlen(p) + 1);
					a.values.push_back(std::string(arff_trim(t.data())));
				} else {
					const std::string tn = upper(tv);
					if (tn != "REAL" && tn != "NUMERIC" && tn != "INTEGER")
						die("unsupported attribute type '" + std::string(tv) + "' ('" + a.name + "') - this standalone loads numeric features and one nominal class only");
				}
				attrs.push_back(std::move(a));
			}
		} else die("unexpected line before @DATA: '" + std::string(line) + "'");
	}
	fclose(f);
	if (!reading_data) die("no @DATA section found");
	if (records.empty()) die("no data rows found");

	// class-major stable sort (mirrors sortRecordsByClass), then widen to double
	ArffData d;
	d.n_features = n_features;
	d.n_classes = n_classes;
	d.class_size.assign(n_classes, 0);
	for (const Record &r : records) {
		if (r.cls < 0 || r.cls >= (int)n_classes) die("record with invalid class id");
		d.class_size[(size_t)r.cls]++;
	}
	d.data.resize((size_t)records.size() * n_features);
	size_t idx = 0;
	for (unsigned int c = 0; c < n_classes; c++)
		for (const Record &r : records)
			if (r.cls == (int)c)
				for (unsigned int ff = 0; ff < n_features; ff++)
					d.data[idx++] = (double)r.feat[ff];
	return d;
}

// ---------------------------------------------------------------- scaling
// Per-feature scaling to [0,1] over the WHOLE data set (the reference's
// load-time scaling; --scaler void skips it).
static void scale_to01(ArffData &d) {
	unsigned int samples = 0;
	for (unsigned int c = 0; c < d.n_classes; c++) samples += d.class_size[c];
	for (unsigned int f = 0; f < d.n_features; f++) {
		double mn = 0, mx = 0; bool first = true;
		size_t idx = f;
		for (unsigned int p = 0; p < samples; p++, idx += d.n_features) {
			const double v = d.data[idx];
			if (first) { mn = mx = v; first = false; }
			else { if (v > mx) mx = v; if (v < mn) mn = v; }
		}
		idx = f;
		for (unsigned int p = 0; p < samples; p++, idx += d.n_features)
			d.data[idx] = (mx > mn) ? (d.data[idx] - mn) / (mx - mn) : 0.0;
	}
}

// --------------------------------------------------------------- splitting
// Depth 0: the paper's RR(1, train%, test%) splitter — a random,
// class-stratified, non-overlapping train/test split. The fill/probe/order
// logic reproduces the reference splitter's PRNG consumption exactly.
// Depth 1 (wrapper configuration only): deterministic k-fold CV of the
// depth-0 train part, sliced positionally per class.
struct Split {
	// absolute sample indices within each class, ascending
	std::vector<std::vector<unsigned int>> train, test; // [class][i]
};

static void rr_split_class(unsigned int n, unsigned int perctrain, unsigned int perctest,
                           std::vector<unsigned int> &train, std::vector<unsigned int> &test) {
	const unsigned char id_empty = 0, id_train = 1, id_test = 2;
	std::vector<unsigned char> mark(n, id_empty);
	const unsigned int trsiz = (n * perctrain) / 100;
	const unsigned int tesiz = (n * perctest) / 100;
	auto fill_randomly = [&](unsigned char from, unsigned char to, unsigned int count) {
		for (unsigned int i = 0; i < count; i++) {
			unsigned int piv = (unsigned int)(ss_rand() % (int)n);
			while (mark[piv] != from) { piv++; if (piv > n - 1) piv = 0; }
			mark[piv] = to;
		}
	};
	auto fill = [&](unsigned char from, unsigned char to) {
		for (unsigned int i = 0; i < n; i++) if (mark[i] == from) mark[i] = to;
	};
	if (perctrain <= 50) fill_randomly(id_empty, id_train, trsiz);
	else { fill(id_empty, id_train); fill_randomly(id_train, id_empty, n - trsiz); }
	if (perctrain + perctest == 100) fill(id_empty, id_test);
	else if (perctest <= (100 - perctrain) / 2) fill_randomly(id_empty, id_test, tesiz);
	else { fill(id_empty, id_test); fill_randomly(id_test, id_empty, n - trsiz - tesiz); }
	train.clear(); test.clear();
	for (unsigned int i = 0; i < n; i++) {
		if (mark[i] == id_train) train.push_back(i);
		else if (mark[i] == id_test) test.push_back(i);
	}
}

// k-fold CV folds of the depth-0 train part; fold sizes follow the
// reference recurrence (remaining/remaining_folds per fold, per class)
static std::vector<Split> cv_folds(const Split &outer, unsigned int kfold, unsigned int n_classes) {
	std::vector<Split> folds(kfold);
	for (unsigned int c = 0; c < n_classes; c++) {
		const std::vector<unsigned int> &tr = outer.train[c];
		unsigned int remaining = (unsigned int)tr.size(), od = kfold, start = 0;
		for (unsigned int k = 0; k < kfold; k++) {
			const unsigned int tcs = remaining / od;
			folds[k].train.resize(n_classes); folds[k].test.resize(n_classes);
			for (unsigned int i = 0; i < (unsigned int)tr.size(); i++) {
				if (i >= start && i < start + tcs) folds[k].test[c].push_back(tr[i]);
				else folds[k].train[c].push_back(tr[i]);
			}
			start += tcs; remaining -= tcs; od--;
		}
	}
	return folds;
}

// ----------------------------------------------------------------- subset
// Feature-subset representation with the reference's sign convention:
// membership = positive marker, regardless of direction mode; in backward
// mode the meaning of the raw select/deselect operations inverts.
struct Subset {
	unsigned int n;
	std::vector<signed char> bin; // +-1 selected/deselected, +-3 traversal
	bool forward = true;
	explicit Subset(unsigned int n_) : n(n_), bin(n_, -1) {}
	signed char id_sel() const { return forward ? 1 : -1; }
	signed char id_desel() const { return forward ? -1 : 1; }
	signed char id_traverse() const { return forward ? 3 : -3; }
	void set_forward_mode(bool fwd) { forward = fwd; }
	void deselect_all() { for (unsigned int i = 0; i < n; i++) bin[i] = id_desel(); }
	void select_raw(unsigned int f) { bin[f] = id_sel(); }
	void deselect_raw(unsigned int f) { bin[f] = id_desel(); }
	bool member(unsigned int f) const { return bin[f] > 0; }
	unsigned int get_d() const { unsigned int d = 0; for (unsigned int i = 0; i < n; i++) if (bin[i] > 0) d++; return d; }
	void members(std::vector<unsigned int> &out) const {
		out.clear();
		for (unsigned int i = 0; i < n; i++) if (bin[i] > 0) out.push_back(i);
	}
	void copy_members_from(const Subset &s) { // stateless_copy: hard +-1, mode-independent
		for (unsigned int i = 0; i < n; i++) bin[i] = s.bin[i] > 0 ? 1 : -1;
	}
	void make_random_subset(unsigned int d) { // reference PRNG consumption: d draws + linear probing
		for (unsigned int i = 0; i < n; i++) bin[i] = id_desel();
		for (unsigned int i = 0; i < d; i++) {
			unsigned int piv = (unsigned int)(ss_rand() % (int)n);
			while (bin[piv] != id_desel()) { piv++; if (piv > n - 1) piv = 0; }
			bin[piv] = id_sel();
		}
	}
};

// ---------------------------------------------------------------- criteria
struct Criterion {
	virtual ~Criterion() {}
	virtual bool evaluate(double &result, const Subset &sub) = 0;
};

// (a) the paper's wrapper criterion: k-NN accuracy, mean over the CV folds
// of the current outer split. Distance = Euclidean over the selected
// features (ascending), neighbour list and vote rule as in the reference.
struct WrapperKnn : Criterion {
	const ArffData &d;
	const std::vector<Split> &folds;    // evaluation splits (CV folds, or the outer split alone)
	unsigned int k;
	std::vector<size_t> class_offset;   // into d.data, per class
	std::vector<unsigned int> feats;    // scratch: selected features
	struct Neighbour { double value; unsigned int cls; };
	std::vector<Neighbour> nns;         // descending by distance; closest LAST
	std::vector<double> scores;

	WrapperKnn(const ArffData &d_, const std::vector<Split> &folds_, unsigned int k_) : d(d_), folds(folds_), k(k_) {
		class_offset.resize(d.n_classes);
		size_t off = 0;
		for (unsigned int c = 0; c < d.n_classes; c++) { class_offset[c] = off; off += (size_t)d.class_size[c] * d.n_features; }
	}
	const double *pattern(unsigned int cls, unsigned int i) const {
		return &d.data[class_offset[cls] + (size_t)i * d.n_features];
	}
	double distance(const double *p1, const double *p2) const {
		double result = 0;
		for (size_t i = 0; i < feats.size(); i++) {
			const unsigned int fi = feats[i];
			const double tmp = p1[fi] - p2[fi];
			result += tmp * tmp;
		}
		return sqrt(result);
	}
	// mirrors Classifier_kNN::sort_in (descending list, closest last, cap)
	void sort_in(double value, unsigned int cls, size_t max_size) {
		size_t pos = 0;
		while (pos != nns.size() && value < nns[pos].value) pos++;
		if (nns.size() < max_size || pos != 0) {
			nns.insert(nns.begin() + (long)pos, Neighbour{value, cls});
			if (nns.size() > max_size) nns.erase(nns.begin());
		}
	}
	bool classify(unsigned int &cls, const double *p, const Split &split) {
		const size_t max_size = (size_t)(k - 1) * d.n_classes + 1; // tie-avoidance cap, as in the reference
		nns.clear();
		for (unsigned int c = 0; c < d.n_classes; c++)
			for (unsigned int i : split.train[c])
				sort_in(distance(p, pattern(c, i)), c, max_size);
		scores.assign(d.n_classes, 0.0);
		unsigned int taken = 0;
		for (size_t it = nns.size(); it-- > 0 && taken < k; taken++) scores[nns[it].cls] += 1.0;
		if (taken == 0) return false;
		for (unsigned int c = 0; c < d.n_classes; c++) scores[c] /= (double)taken;
		cls = 0;
		for (unsigned int c = 1; c < d.n_classes; c++) if (scores[c] > scores[cls]) cls = c; // first maximum wins
		return true;
	}
	bool evaluate(double &result, const Subset &sub) override {
		sub.members(feats);
		if (feats.empty()) return false;
		result = 0.0;
		double cnt = 0.0;
		for (const Split &split : folds) {
			unsigned long long count = 0, correct = 0;
			unsigned int clstmp;
			for (unsigned int c_test = 0; c_test < d.n_classes; c_test++)
				for (unsigned int i : split.test[c_test]) {
					if (!classify(clstmp, pattern(c_test, i), split)) return false;
					if (clstmp == c_test) correct++;
					count++;
				}
			result += (count > 0) ? (double)correct / (double)count : 0.0;
			cnt += 1.0;
		}
		result /= cnt;
		return true;
	}
};

// (b) the paper's filter criterion: multinomial Bhattacharyya distance,
// model learned once on the outer split's train part (Laplace-smoothed
// term probabilities per class; class priors from full class sizes).
// Singleton subsets score by the cached Individual Bhattacharyya, exactly
// as in the reference.
struct MultinomBhattacharyya : Criterion {
	const ArffData &d;
	unsigned int n, classes;
	std::vector<double> Nsuminclass; // [class][feature], train-part term counts
	std::vector<double> Pc;          // class priors (full class sizes)
	std::vector<double> Pc_d;        // per-class share of total train term mass
	std::vector<double> theta;       // [class][narrowed feature]
	std::vector<double> IB;          // per-feature individual Bhattacharyya
	std::vector<unsigned int> index; // narrow map
	unsigned int allpatterns = 0;
	double doc_avg_length = 0;
	bool IB_computed = false;
	std::vector<unsigned int> feats;

	MultinomBhattacharyya(const ArffData &d_, const Split &outer) : d(d_), n(d_.n_features), classes(d_.n_classes) {
		Nsuminclass.assign((size_t)n * classes, 0.0);
		Pc.resize(classes); Pc_d.resize(classes);
		theta.resize((size_t)n * classes);
		IB.resize(n);
		index.resize(n);
		// class priors from FULL class sizes (the reference convention)
		unsigned int class_size_sum = 0;
		for (unsigned int c = 0; c < classes; c++) class_size_sum += d.class_size[c];
		for (unsigned int c = 0; c < classes; c++) Pc[c] = (double)d.class_size[c] / (double)class_size_sum;
		// sufficient statistics from the train part, block (=ascending) order
		std::vector<size_t> class_offset(classes);
		size_t off = 0;
		for (unsigned int c = 0; c < classes; c++) { class_offset[c] = off; off += (size_t)d.class_size[c] * n; }
		size_t wCV = 0;
		for (unsigned int c = 0; c < classes; c++) {
			for (unsigned int i : outer.train[c]) {
				const double *p = &d.data[class_offset[c] + (size_t)i * n];
				for (unsigned int f = 0; f < n; f++) Nsuminclass[wCV + f] += p[f];
			}
			allpatterns += (unsigned int)outer.train[c].size();
			wCV += n;
		}
		double total_sum_length = 0;
		wCV = 0;
		for (unsigned int c = 0; c < classes; c++) {
			for (unsigned int f = 0; f < n; f++) total_sum_length += Nsuminclass[wCV + f];
			wCV += n;
		}
		if (total_sum_length == 0) {
			for (unsigned int c = 0; c < classes; c++) Pc_d[c] = 1.0 / (double)classes;
		} else {
			wCV = 0;
			for (unsigned int c = 0; c < classes; c++) {
				double class_sum_length = 0;
				for (unsigned int f = 0; f < n; f++) class_sum_length += Nsuminclass[wCV + f];
				Pc_d[c] = class_sum_length / total_sum_length;
				wCV += n;
			}
		}
	}
	// theta over the current narrow map of width dd (mirrors compute_theta)
	void compute_theta(unsigned int dd) {
		double total_sum_length = 0;
		size_t wCV = 0, wCd = 0;
		for (unsigned int c = 0; c < classes; c++) {
			double class_sum_length = 0;
			for (unsigned int f = 0; f < dd; f++) class_sum_length += Nsuminclass[wCV + index[f]];
			total_sum_length += class_sum_length;
			for (unsigned int f = 0; f < dd; f++)
				theta[wCd++] = (1.0 + Nsuminclass[wCV + index[f]]) / ((double)dd + class_sum_length);
			wCV += n;
		}
		doc_avg_length = total_sum_length / (double)allpatterns;
	}
	void compute_IB() { // mirrors compute_IB (theta at full width required)
		for (unsigned int f = 0; f < n; f++) IB[f] = 0.0;
		for (unsigned int f = 0; f < n; f++) {
			double value = 0.0; unsigned int combs = 0;
			size_t wCV1 = 0;
			for (unsigned int c1 = 0; c1 < classes; c1++) {
				size_t wCV2 = wCV1 + n;
				for (unsigned int c2 = c1 + 1; c2 < classes; c2++) {
					const double thetasum = sqrt(theta[wCV1 + f] * theta[wCV2 + f]) + sqrt((1.0 - theta[wCV1 + f]) * (1.0 - theta[wCV2 + f]));
					value += ((-doc_avg_length) * log(thetasum)) * Pc_d[c1] * Pc_d[c2];
					combs++;
					wCV2 += n;
				}
				wCV1 += n;
			}
			IB[f] = value / (double)combs;
		}
	}
	bool evaluate(double &result, const Subset &sub) override {
		sub.members(feats);
		const unsigned int dd = (unsigned int)feats.size();
		if (dd == 0) return false;
		if (dd == 1) {
			if (!IB_computed) {
				for (unsigned int i = 0; i < n; i++) index[i] = i; // denarrow
				compute_theta(n);
				compute_IB();
				IB_computed = true;
			}
			result = IB[feats[0]];
		} else {
			for (unsigned int i = 0; i < dd; i++) index[i] = feats[i]; // narrow_to
			compute_theta(dd);
			double value = 0.0;
			for (unsigned int c1 = 0; c1 < classes; c1++)
				for (unsigned int c2 = c1 + 1; c2 < classes; c2++) {
					const double *t1 = &theta[(size_t)c1 * dd];
					const double *t2 = &theta[(size_t)c2 * dd];
					double thetasum = 0.0;
					for (unsigned int f = 0; f < dd; f++) thetasum += sqrt(t1[f] * t2[f]);
					value += log(thetasum) * Pc[c1] * Pc[c2];
				}
			result = (-doc_avg_length) * value;
		}
		return true;
	}
};

// counts every criterion evaluation (the reference's counting decorator)
struct CountingCriterion : Criterion {
	Criterion &inner;
	unsigned long long count = 0;
	explicit CountingCriterion(Criterion &c) : inner(c) {}
	bool evaluate(double &result, const Subset &sub) override { count++; return inner.evaluate(result, sub); }
};

// ------------------------------------------------- online statistics
// The per-feature statistics engine (Result_Weighter_Sampled): buffered
// evaluations are z-standardized per batch and folded into adaptive-rate
// per-feature accumulators; score(f) = m_is(f) - m_isnot(f).
struct Weighter {
	struct FeatureStat { double m_is = 0, m_isnot = 0, v_is = 0; unsigned int n_is = 0, n_isnot = 0; };
	struct BatchEntry { double value; std::vector<unsigned int> selected; };
	unsigned int n = 0;
	unsigned int horizon;
	bool frozen = false;
	std::vector<FeatureStat> fs;
	std::vector<BatchEntry> batch;
	std::vector<char> mark;

	explicit Weighter(unsigned int horizon_) : horizon(horizon_ < 1 ? 1 : horizon_) {}
	void reset(unsigned int n_) {
		n = n_;
		fs.assign(n, FeatureStat());
		mark.assign(n, 0);
		batch.clear();
		frozen = false;
	}
	void add(double value, const Subset &sub) {
		if (frozen) return;
		if (n == 0) reset(sub.n);
		BatchEntry e;
		e.value = value;
		sub.members(e.selected);
		batch.push_back(std::move(e));
	}
	void flush_batch() {
		if (frozen) { batch.clear(); return; }
		const size_t B = batch.size();
		if (B == 0) return;
		double mu = 0.0;
		for (size_t i = 0; i < B; i++) mu += batch[i].value;
		mu /= (double)B;
		double var = 0.0;
		for (size_t i = 0; i < B; i++) { const double dv = batch[i].value - mu; var += dv * dv; }
		var /= (double)B;
		const double sd = sqrt(var);
		const bool degenerate = !(sd > 1e-12);
		for (size_t i = 0; i < B; i++) {
			const BatchEntry &e = batch[i];
			const double z = degenerate ? 0.0 : (e.value - mu) / sd;
			for (unsigned int f : e.selected) mark[f] = 1;
			for (unsigned int j = 0; j < n; j++) {
				FeatureStat &s = fs[j];
				if (mark[j]) {
					s.n_is++;
					const double a = 1.0 / (double)(s.n_is < horizon ? s.n_is : horizon);
					s.m_is += a * (z - s.m_is);
					s.v_is += a * (z * z - s.v_is);
				} else {
					s.n_isnot++;
					const double a = 1.0 / (double)(s.n_isnot < horizon ? s.n_isnot : horizon);
					s.m_isnot += a * (z - s.m_isnot);
				}
			}
			for (unsigned int f : e.selected) mark[f] = 0;
		}
		batch.clear();
	}
	void freeze() { flush_batch(); frozen = true; }
	double score(unsigned int f) const { return fs[f].m_is - fs[f].m_isnot; }
	unsigned int count_is(unsigned int f) const { return fs[f].n_is; }
	unsigned int count_isnot(unsigned int f) const { return fs[f].n_isnot; }
};

// -------------------------------------------------- budgeted sampled step
// The sADD/sRMV operator pair (Sequential_Step_Sampled): PRNG consumption,
// proposal construction, tie rules and instrumentation reproduce the
// reference implementation exactly.
enum SamplerMode { SAMPLER_SOFTMAX, SAMPLER_UNIFORM, SAMPLER_TOPK };

struct SampledStep {
	Weighter &stats;
	unsigned int cap;             // forward budget y_f
	unsigned int cap_backward;    // backward budget y_b (0 = full sweeps)
	double cap_frac;              // >0: ceil(frac*pool) overrides cap (forward)
	double explore;               // exploration-floor fraction
	double tau;                   // softmax temperature (<=0 = robust auto)
	SamplerMode sampler;
	// instrumentation (the reference's F/B order diagnostics)
	unsigned int count_forward = 0, count_backward = 0;
	double order_forward = 0.0, order_backward = 0.0;
	unsigned long long all_evals = 0;

	SampledStep(Weighter &stats_, unsigned int cap_, unsigned int cap_backward_, double cap_frac_,
	            double explore_, double tau_, SamplerMode sampler_)
		: stats(stats_), cap(cap_), cap_backward(cap_backward_), cap_frac(cap_frac_),
		  explore(explore_ < 0.0 ? 0.0 : (explore_ > 1.0 ? 1.0 : explore_)), tau(tau_), sampler(sampler_) {}

	void note_step(bool forward, unsigned int order, unsigned int cnt, unsigned int pool, std::ostream &os) {
		const double r = (double)order / (double)cnt;
		if (forward) {
			count_forward++;
			order_forward = ((count_forward - 1) * order_forward + r) / count_forward;
			os << std::endl << "F order = " << r << "(" << order << ") avgorder " << order_forward << " count = " << count_forward << " all_evals = " << all_evals << " pool = " << pool << " proposed = " << cnt << std::endl << std::flush;
		} else {
			count_backward++;
			order_backward = ((count_backward - 1) * order_backward + r) / count_backward;
			os << std::endl << "B order = " << r << "(" << order << ") avgorder " << order_backward << " count = " << count_backward << " all_evals = " << all_evals << " pool = " << pool << " proposed = " << cnt << std::endl << std::flush;
		}
	}

	// classic exhaustive sweep over the current direction's candidate pool
	// (g=1): the fallback for pool<=cap and for cap==0
	bool full_sweep(double &result, Subset &sub, CountingCriterion &crit, unsigned int pool, std::ostream &os) {
		const bool forward = sub.forward;
		double bestval = 0;
		unsigned int order = 0, cnt = 0;
		bool best_available = false;
		unsigned int bestf = sub.n;
		const signed char cand_from = sub.id_desel();      // traversal pool marker
		const signed char cand_mark = sub.id_traverse();
		for (unsigned int f = 0; f < sub.n; f++) {
			if (sub.bin[f] != cand_from) continue;
			sub.bin[f] = cand_mark;                        // candidate = sub +- {f}
			double val;
			if (!crit.evaluate(val, sub)) return false;
			all_evals++;
			stats.add(val, sub);
			cnt++;
			if (!best_available || val > bestval) { bestval = val; order = cnt; bestf = f; best_available = true; }
			sub.bin[f] = cand_from;
		}
		if (!best_available) return false;
		stats.flush_batch();
		// apply the winner (equivalent to the engine's best-subset copy)
		sub.bin[bestf] = sub.id_sel();
		result = bestval;
		note_step(forward, order, cnt, pool, os);
		return true;
	}

	bool sampled_forward(double &result, Subset &sub, CountingCriterion &crit, unsigned int pool, unsigned int want, std::ostream &os) {
		const unsigned int n = sub.n;
		std::vector<unsigned int> freef;
		freef.reserve(pool);
		for (unsigned int f = 0; f < n; f++) if (!sub.member(f)) freef.push_back(f);
		std::vector<unsigned int> prop;
		prop.reserve(want);
		std::vector<char> taken(n, 0);
		if (sampler == SAMPLER_TOPK) {
			std::vector<unsigned int> byscore(freef);
			std::partial_sort(byscore.begin(), byscore.begin() + want, byscore.end(),
				[this](unsigned int a, unsigned int b) { const double sa = stats.score(a), sb = stats.score(b); return sa > sb || (sa == sb && a < b); });
			for (unsigned int i = 0; i < want; i++) prop.push_back(byscore[i]);
		} else {
			unsigned int u = (sampler == SAMPLER_UNIFORM) ? want : (unsigned int)(explore * (double)want + 0.5);
			if (u > want) u = want;
			if (u > 0) {
				std::vector<unsigned int> bycnt(freef);
				const unsigned int le = (sampler == SAMPLER_UNIFORM) ? pool : std::min<unsigned int>(pool, std::max<unsigned int>(4 * u, u));
				if (le < pool)
					std::partial_sort(bycnt.begin(), bycnt.begin() + le, bycnt.end(),
						[this](unsigned int a, unsigned int b) { const auto ca = stats.count_is(a), cb = stats.count_is(b); return ca < cb || (ca == cb && a < b); });
				for (unsigned int k = 0; k < u; k++) {
					const unsigned int j = k + (unsigned int)(frand() * (double)(le - k));
					std::swap(bycnt[k], bycnt[j]);
					prop.push_back(bycnt[k]);
					taken[bycnt[k]] = 1;
				}
			}
			const unsigned int rest = want - u;
			if (rest > 0) {
				std::vector<unsigned int> cand;
				cand.reserve(pool - u);
				for (size_t i = 0; i < freef.size(); i++) if (!taken[freef[i]]) cand.push_back(freef[i]);
				std::vector<double> sc(cand.size());
				for (size_t i = 0; i < cand.size(); i++) sc[i] = stats.score(cand[i]);
				double t = tau;
				if (t <= 0.0) {
					std::vector<double> tmp(sc);
					const size_t q1 = tmp.size() / 4, q3 = (3 * tmp.size()) / 4;
					std::nth_element(tmp.begin(), tmp.begin() + q1, tmp.end());
					const double a = tmp[q1];
					std::nth_element(tmp.begin(), tmp.begin() + q3, tmp.end());
					const double b = tmp[q3];
					t = (b - a) / 1.349;
					if (!(t > 1e-9)) t = 1e-9;
				}
				const double smax = *std::max_element(sc.begin(), sc.end());
				std::vector<double> w(sc.size());
				double W = 0.0;
				for (size_t i = 0; i < sc.size(); i++) { w[i] = std::exp((sc[i] - smax) / t); W += w[i]; }
				for (unsigned int k = 0; k < rest; k++) {
					const double r = frand() * W;
					size_t pick = w.size();
					double acc = 0.0;
					for (size_t i = 0; i < w.size(); i++) {
						if (w[i] <= 0.0) continue;
						acc += w[i];
						if (r < acc) { pick = i; break; }
					}
					if (pick == w.size()) { for (size_t i = w.size(); i-- > 0;) if (w[i] > 0.0) { pick = i; break; } }
					if (pick == w.size()) { // complete-underflow guard: tau->0 limit = argmax(score)
						for (size_t i = 0; i < w.size(); i++) if (w[i] >= 0.0 && (pick == w.size() || sc[i] > sc[pick])) pick = i;
					}
					prop.push_back(cand[pick]);
					W -= w[pick];
					w[pick] = -1.0;
				}
			}
		}
		// evaluate the batch: candidate i = S + {prop[i]}
		double bestval = 0;
		unsigned int bestf = n, order = 0, cnt = 0;
		for (size_t i = 0; i < prop.size(); i++) {
			const unsigned int f = prop[i];
			sub.select_raw(f);
			double val;
			if (!crit.evaluate(val, sub)) return false;
			all_evals++;
			stats.add(val, sub);
			cnt++;
			if (bestf == n || val > bestval) { bestval = val; bestf = f; order = cnt; }
			sub.deselect_raw(f);
		}
		stats.flush_batch();
		if (bestf == n) return false;
		sub.select_raw(bestf);
		result = bestval;
		note_step(true, order, cnt, pool, os);
		return true;
	}

	bool sampled_backward(double &result, Subset &sub, CountingCriterion &crit, unsigned int pool, unsigned int want, std::ostream &os) {
		const unsigned int n = sub.n;
		std::vector<unsigned int> memb;
		memb.reserve(pool);
		for (unsigned int f = 0; f < n; f++) if (sub.member(f)) memb.push_back(f);
		std::vector<unsigned int> prop;
		prop.reserve(want);
		std::vector<char> taken(n, 0);
		if (sampler == SAMPLER_TOPK) {
			std::vector<unsigned int> byscore(memb);
			std::partial_sort(byscore.begin(), byscore.begin() + want, byscore.end(),
				[this](unsigned int a, unsigned int b) { const double sa = stats.score(a), sb = stats.score(b); return sa < sb || (sa == sb && a < b); });
			for (unsigned int i = 0; i < want; i++) prop.push_back(byscore[i]);
		} else {
			unsigned int u = (sampler == SAMPLER_UNIFORM) ? want : (unsigned int)(explore * (double)want + 0.5);
			if (u > want) u = want;
			if (u > 0) {
				std::vector<unsigned int> bycnt(memb);
				const unsigned int le = (sampler == SAMPLER_UNIFORM) ? pool : std::min<unsigned int>(pool, std::max<unsigned int>(4 * u, u));
				if (le < pool)
					std::partial_sort(bycnt.begin(), bycnt.begin() + le, bycnt.end(),
						[this](unsigned int a, unsigned int b) { const auto ca = stats.count_isnot(a), cb = stats.count_isnot(b); return ca < cb || (ca == cb && a < b); });
				for (unsigned int k = 0; k < u; k++) {
					const unsigned int j = k + (unsigned int)(frand() * (double)(le - k));
					std::swap(bycnt[k], bycnt[j]);
					prop.push_back(bycnt[k]);
					taken[bycnt[k]] = 1;
				}
			}
			const unsigned int rest = want - u;
			if (rest > 0) {
				std::vector<unsigned int> cand;
				cand.reserve(pool - u);
				for (size_t i = 0; i < memb.size(); i++) if (!taken[memb[i]]) cand.push_back(memb[i]);
				std::vector<double> sc(cand.size());
				for (size_t i = 0; i < cand.size(); i++) sc[i] = stats.score(cand[i]);
				double t = tau;
				if (t <= 0.0) {
					std::vector<double> tmp(sc);
					const size_t q1 = tmp.size() / 4, q3 = (3 * tmp.size()) / 4;
					std::nth_element(tmp.begin(), tmp.begin() + q1, tmp.end());
					const double a = tmp[q1];
					std::nth_element(tmp.begin(), tmp.begin() + q3, tmp.end());
					const double b = tmp[q3];
					t = (b - a) / 1.349;
					if (!(t > 1e-9)) t = 1e-9;
				}
				const double smin = *std::min_element(sc.begin(), sc.end());
				std::vector<double> w(sc.size());
				double W = 0.0;
				for (size_t i = 0; i < sc.size(); i++) { w[i] = std::exp((smin - sc[i]) / t); W += w[i]; }
				for (unsigned int k = 0; k < rest; k++) {
					const double r = frand() * W;
					size_t pick = w.size();
					double acc = 0.0;
					for (size_t i = 0; i < w.size(); i++) {
						if (w[i] <= 0.0) continue;
						acc += w[i];
						if (r < acc) { pick = i; break; }
					}
					if (pick == w.size()) { for (size_t i = w.size(); i-- > 0;) if (w[i] > 0.0) { pick = i; break; } }
					if (pick == w.size()) { // complete-underflow guard: tau->0 limit = argmin(score)
						for (size_t i = 0; i < w.size(); i++) if (w[i] >= 0.0 && (pick == w.size() || sc[i] < sc[pick])) pick = i;
					}
					prop.push_back(cand[pick]);
					W -= w[pick];
					w[pick] = -1.0;
				}
			}
		}
		// evaluate the batch in place: candidate i = S - {prop[i]}
		double bestval = 0;
		unsigned int bestf = n, order = 0, cnt = 0;
		for (size_t i = 0; i < prop.size(); i++) {
			const unsigned int f = prop[i];
			sub.select_raw(f);   // backward mode: removes f
			double val;
			if (!crit.evaluate(val, sub)) return false;
			all_evals++;
			stats.add(val, sub);
			cnt++;
			if (bestf == n || val > bestval) { bestval = val; bestf = f; order = cnt; }
			sub.deselect_raw(f); // backward mode: restores f
		}
		stats.flush_batch();
		if (bestf == n) return false;
		sub.select_raw(bestf);
		result = bestval;
		note_step(false, order, cnt, pool, os);
		return true;
	}

	bool evaluate_candidates(double &result, Subset &sub, CountingCriterion &crit, std::ostream &os) {
		if (stats.n != sub.n) stats.reset(sub.n);
		const bool forward = sub.forward;
		const unsigned int pool = forward ? (sub.n - sub.get_d()) : sub.get_d();
		if (pool == 0) return false;
		if (forward) {
			unsigned int c = cap;
			if (cap_frac > 0.0) { c = (unsigned int)std::ceil(cap_frac * (double)pool); if (c < 1) c = 1; }
			if (c == 0 || pool <= c) return full_sweep(result, sub, crit, pool, os);
			return sampled_forward(result, sub, crit, pool, c, os);
		}
		if (cap_backward == 0 || pool <= cap_backward) return full_sweep(result, sub, crit, pool, os);
		return sampled_backward(result, sub, crit, pool, cap_backward, os);
	}

	// one sequential Step: temporary direction-mode flip, as in the reference
	bool Step(bool forward, double &result, Subset &sub, CountingCriterion &crit, std::ostream &os) {
		const bool mode_change = (sub.forward != forward);
		if (mode_change) sub.set_forward_mode(forward);
		const bool success = evaluate_candidates(result, sub, crit, os);
		if (mode_change) sub.set_forward_mode(!forward);
		return success;
	}
};

// ------------------------------------------------------------------ SFFS
// Sequential Forward Floating Search over the substituted budgeted
// operators = sSFFS. Loop structure, per-size incumbents, MAXCRIT tie
// rules and the frontier restriction mirror the reference Search_SFFS.
// Solution announcements are emitted as the reference's --progress json
// lines: {"event":"solution","value":V,"d":D,"features":[...]}.
static void emit_solution(std::ostream &os, double value, const Subset &sub) {
	std::ostringstream sos;
	sos << "{\"event\":\"solution\",\"value\":" << value << ",\"d\":" << sub.get_d() << ",\"features\":[";
	bool first = true;
	for (unsigned int f = 0; f < sub.n; f++) if (sub.member(f)) { sos << (first ? "" : ",") << f; first = false; }
	sos << "]}" << std::endl;
	os << sos.str() << std::flush;
}

struct SffsResult {
	double value = 0;
	std::vector<unsigned int> features;
};

static bool sffs_search(unsigned int target_d, unsigned int delta, Subset &sub, CountingCriterion &crit,
                        SampledStep &step, std::ostream &os, SffsResult &out) {
	const unsigned int n = sub.n;
	struct OneSubset { double critvalue = 0; bool present = false; std::vector<signed char> bin; };
	std::vector<OneSubset> bsubs(n);
	Subset pivotsub(n), maxcritsub(n);
	double maxcritval = 0;
	bool havemax = false;
	unsigned int d_max = 0;

	sub.set_forward_mode(true);
	sub.deselect_all();
	unsigned int d = 0;
	double result = 0;

	const unsigned int forward_thr = (target_d > 0 && delta > 0 && target_d + delta < n) ? target_d + delta : n;
	while (d + 1 <= forward_thr) {
		// unconditional forward step
		if (!step.Step(true, result, sub, crit, os)) return false;
		d = sub.get_d();
		if (!havemax || result > maxcritval || (result == maxcritval && d < d_max)) {
			maxcritsub.copy_members_from(sub); maxcritval = result; d_max = d; havemax = true;
			emit_solution(os, maxcritval, maxcritsub);
		}
		if (!bsubs[d - 1].present) {
			bsubs[d - 1].present = true;
			bsubs[d - 1].bin.assign(sub.bin.begin(), sub.bin.end());
			bsubs[d - 1].critvalue = result;
			emit_solution(os, result, sub);
		} else if (result > bsubs[d - 1].critvalue) {
			bsubs[d - 1].bin.assign(sub.bin.begin(), sub.bin.end());
			bsubs[d - 1].critvalue = result;
			emit_solution(os, result, sub);
		}
		pivotsub.copy_members_from(sub);
		// conditional backtracking
		bool backtrack = true;
		while (backtrack && d >= 2) {
			if (!step.Step(false, result, sub, crit, os)) return false;
			d = sub.get_d();
			if (!havemax || result > maxcritval || (result == maxcritval && d < d_max)) {
				maxcritsub.copy_members_from(sub); maxcritval = result; d_max = d; havemax = true;
				emit_solution(os, maxcritval, maxcritsub);
			}
			if (!bsubs[d - 1].present) {
				bsubs[d - 1].present = true;
				bsubs[d - 1].bin.assign(sub.bin.begin(), sub.bin.end());
				bsubs[d - 1].critvalue = result;
				emit_solution(os, result, sub);
			} else if (result > bsubs[d - 1].critvalue) {
				bsubs[d - 1].bin.assign(sub.bin.begin(), sub.bin.end());
				bsubs[d - 1].critvalue = result;
				pivotsub.copy_members_from(sub);
				emit_solution(os, result, sub);
			} else backtrack = false;
		}
		sub.bin.assign(pivotsub.bin.begin(), pivotsub.bin.end());
		d = sub.get_d();
	}
	if (target_d > 0) {
		if (!bsubs[target_d - 1].present) return false;
		out.value = bsubs[target_d - 1].critvalue;
		out.features.clear();
		for (unsigned int f = 0; f < n; f++) if (bsubs[target_d - 1].bin[f] > 0) out.features.push_back(f);
	} else {
		if (!havemax) return false;
		out.value = maxcritval;
		out.features.clear();
		for (unsigned int f = 0; f < n; f++) if (maxcritsub.member(f)) out.features.push_back(f);
	}
	return true;
}

// ------------------------------------------------------------------- main
struct Config {
	std::string data;
	unsigned int rr_train = 50, rr_test = 50;
	unsigned int cv_folds = 0;          // 0 = no inner CV (criterion runs on the outer split)
	std::string scaler = "void";        // void | to01
	std::string criterion;              // wrapper-knn | multinom-bhattacharyya
	unsigned int knn_k = 1;
	unsigned int target_d = 0;
	unsigned int sffs_delta = 0;        // 0 = unrestricted floating frontier
	unsigned int step_cap = 100;
	unsigned int step_cap_backward = 0; // 0 = full backward sweeps (exact floating correction)
	double step_cap_frac = 0.0;
	double step_explore = 0.2;
	double step_tau = 0.0;
	unsigned int step_decay = 100;
	std::string step_sampler = "softmax";
	bool step_frozen = false;
	unsigned long warmup_probes = 0;
	unsigned int warmup_card = 25;
	unsigned int seed = 1;
};

static std::string fmt_double(double v) {
	char buf[48];
	snprintf(buf, sizeof(buf), "%.12g", v);
	return std::string(buf);
}

int main(int argc, char **argv) {
	Config cfg;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		auto next = [&]() -> const char * { if (++i >= argc) die("missing value for " + a); return argv[i]; };
		if (a == "--data") cfg.data = next();
		else if (a == "--rr-train") cfg.rr_train = (unsigned int)atoi(next());
		else if (a == "--rr-test") cfg.rr_test = (unsigned int)atoi(next());
		else if (a == "--cv-folds") cfg.cv_folds = (unsigned int)atoi(next());
		else if (a == "--scaler") cfg.scaler = next();
		else if (a == "--criterion") cfg.criterion = next();
		else if (a == "--knn-k") cfg.knn_k = (unsigned int)atoi(next());
		else if (a == "--target-d") cfg.target_d = (unsigned int)atoi(next());
		else if (a == "--sffs-delta") cfg.sffs_delta = (unsigned int)atoi(next());
		else if (a == "--step-cap") cfg.step_cap = (unsigned int)atoi(next());
		else if (a == "--step-cap-backward") cfg.step_cap_backward = (unsigned int)atoi(next());
		else if (a == "--step-cap-frac") cfg.step_cap_frac = atof(next());
		else if (a == "--step-explore") cfg.step_explore = atof(next());
		else if (a == "--step-tau") cfg.step_tau = atof(next());
		else if (a == "--step-decay") cfg.step_decay = (unsigned int)atoi(next());
		else if (a == "--step-sampler") cfg.step_sampler = next();
		else if (a == "--step-frozen") cfg.step_frozen = true;
		else if (a == "--warmup-probes") cfg.warmup_probes = (unsigned long)atol(next());
		else if (a == "--warmup-card") cfg.warmup_card = (unsigned int)atoi(next());
		else if (a == "--seed") cfg.seed = (unsigned int)atoi(next());
		else if (a == "--help" || a == "-h") {
			std::cout <<
				"ssffs - standalone Stochastic Sequential Forward Floating Search (sSFFS)\n"
				"see the header of ssffs.cpp and README.md for the paper configurations\n"
				"options: --data F.arff --rr-train P --rr-test P [--cv-folds K]\n"
				"         --scaler void|to01 --criterion wrapper-knn|multinom-bhattacharyya\n"
				"         [--knn-k K] --target-d D [--sffs-delta D]\n"
				"         [--step-cap N] [--step-cap-backward N] [--step-cap-frac F]\n"
				"         [--step-explore F] [--step-tau F] [--step-decay N]\n"
				"         [--step-sampler softmax|uniform|topk] [--step-frozen]\n"
				"         [--warmup-probes N] [--warmup-card C] [--seed S]\n";
			return 0;
		}
		else die("unknown option '" + a + "' (see --help)");
	}
	if (cfg.data.empty()) die("--data is required");
	if (cfg.criterion != "wrapper-knn" && cfg.criterion != "multinom-bhattacharyya")
		die("--criterion must be wrapper-knn or multinom-bhattacharyya");
	if (cfg.scaler != "void" && cfg.scaler != "to01") die("--scaler must be void or to01");
	if (cfg.rr_train + cfg.rr_test > 100) die("--rr-train + --rr-test must be <= 100");
	SamplerMode sm = SAMPLER_SOFTMAX;
	if (cfg.step_sampler == "uniform") sm = SAMPLER_UNIFORM;
	else if (cfg.step_sampler == "topk") sm = SAMPLER_TOPK;
	else if (cfg.step_sampler != "softmax") die("--step-sampler must be softmax, uniform or topk");

	// pipeline order matches the reference: seed once; load + scale; make the
	// depth-0 split (the only PRNG consumer before warm-up); build criterion;
	// warm-up probes (one statistics batch); run the search
	ss_srand(cfg.seed);
	ArffData d = load_arff(cfg.data);
	if (cfg.scaler == "to01") scale_to01(d);
	const unsigned int n = d.n_features;
	if (cfg.target_d >= n) die("--target-d must be smaller than the number of features");

	Split outer;
	outer.train.resize(d.n_classes); outer.test.resize(d.n_classes);
	for (unsigned int c = 0; c < d.n_classes; c++)
		rr_split_class(d.class_size[c], cfg.rr_train, cfg.rr_test, outer.train[c], outer.test[c]);

	std::vector<Split> eval_splits; // the splits the criterion evaluates over
	if (cfg.cv_folds > 1) eval_splits = cv_folds(outer, cfg.cv_folds, d.n_classes);
	else eval_splits.push_back(outer);

	std::unique_ptr<Criterion> crit_inner;
	if (cfg.criterion == "wrapper-knn") crit_inner.reset(new WrapperKnn(d, eval_splits, cfg.knn_k));
	else crit_inner.reset(new MultinomBhattacharyya(d, outer));
	CountingCriterion crit(*crit_inner);

	Weighter stats(cfg.step_decay);
	SampledStep step(stats, cfg.step_cap, cfg.step_cap_backward, cfg.step_cap_frac,
	                 cfg.step_explore, cfg.step_tau, sm);

	// warm-up: random probes of fixed cardinality seed the statistics; all
	// probes form ONE batch (flushed once), exactly as in the reference
	if (cfg.warmup_probes > 0) {
		const unsigned int r = cfg.warmup_card < n ? cfg.warmup_card : n - 1;
		Subset probe(n);
		stats.reset(n);
		double pv;
		for (unsigned long i = 0; i < cfg.warmup_probes; i++) {
			probe.make_random_subset(r);
			if (!crit.evaluate(pv, probe)) die("warm-up probe evaluation failed");
			stats.add(pv, probe);
		}
		stats.flush_batch();
		if (cfg.step_frozen) stats.freeze();
	}

	Subset sub(n);
	SffsResult res;
	if (!sffs_search(cfg.target_d, cfg.sffs_delta, sub, crit, step, std::cerr, res))
		die("search not finished");

	std::cout << "{\"value\": " << fmt_double(res.value)
	          << ", \"size\": " << res.features.size() << ", \"features\": [";
	for (size_t i = 0; i < res.features.size(); i++) std::cout << (i ? ", " : "") << res.features[i];
	std::cout << "], \"evaluations\": " << crit.count << "}" << std::endl;
	return 0;
}
