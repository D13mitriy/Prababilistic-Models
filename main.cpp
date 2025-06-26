// lab_pattern_recognition.cpp
// Laboratory work: Probabilistic models in pattern-recognition
// VC-theoretic and simplified bounds for sample size, CSV outputs, visualization support


#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace chrono;


struct Point {
    double x1, x2;
    int label; // 0 for white, 1 for black
};

// Strategy: q(x) = 0 if (x, a) >= theta, else 1
int classify(const Point& x, const vector<double>& a, double theta) {
    double dot = x.x1 * a[0] + x.x2 * a[1];
    return (dot >= theta) ? 0 : 1;
}

// Empirical risk of a strategy (a, theta)
double empirical_risk(const vector<Point>& data, const vector<double>& a, double theta) {
    int errors = 0;
    for (const auto& pt : data)
        if (classify(pt, a, theta) != pt.label)
            ++errors;
    return static_cast<double>(errors) / data.size();
}

// Check linear separability of labeling
bool is_linearly_separable(const vector<Point>& points, const vector<int>& labels) {
    const int steps = 36;
    for (int angle = 0; angle < steps; ++angle) {
        double theta = angle * M_PI / steps;
        double nx = cos(theta), ny = sin(theta);
        vector<double> projections;
        for (auto& pt : points)
            projections.push_back(pt.x1 * nx + pt.x2 * ny);

        vector<pair<double, int>> proj_label;
        for (int i = 0; i < points.size(); ++i)
            proj_label.emplace_back(projections[i], labels[i]);

        sort(proj_label.begin(), proj_label.end());
        for (int split = 0; split < proj_label.size(); ++split) {
            bool good = true;
            int left = proj_label[split].second;
            for (int j = 0; j < proj_label.size(); ++j) {
                int side = j <= split ? left : 1 - left;
                if (proj_label[j].second != side) {
                    good = false;
                    break;
                }
            }
            if (good) return true;
        }
    }
    return false;
}

// Compute VC-dimension for linear classifiers in R^2
int compute_vc_dimension() {
    vector<vector<Point>> configs = {
        {{0,0}},
        {{0,0}, {1,0}},
        {{0,0}, {1,0}, {0,1}},
        {{0,0}, {1,0}, {0,1}, {1,1}}
    };

    for (int n = 1; n <= 4; ++n) {
        const auto& points = configs[n-1];
        bool all_shattered = true;
        for (int mask = 0; mask < (1 << n); ++mask) {
            vector<int> labels(n);
            for (int i = 0; i < n; ++i)
                labels[i] = (mask >> i) & 1;
            if (!is_linearly_separable(points, labels)) {
                all_shattered = false;
                break;
            }
        }
        if (!all_shattered)
            return n - 1;
    }
    return 4;
}

// Compute minimal sample size l from VC bound
int compute_simplified_sample_size(double epsilon, double eta) {
    return static_cast<int>(ceil((1.0 / (2 * epsilon * epsilon)) * log(2.0 / eta)));
    //return static_cast<int>(ceil((log(N) - log(eta))/(2 * pow(epsilon, 2))));
}

// VC-theoretic generalization bound (solved with binary search)
int compute_vc_sample_size(int vc_dim, double epsilon, double eta) {
    int left = 1, right = 100000;
    while (left < right) {
        int l = (left + right) / 2;
        double bound = (1.0 / (epsilon * epsilon)) *
                       (4 * vc_dim * log((2 * M_E * l) / vc_dim) + log(4.0 / eta));
        if (l >= bound) right = l;
        else left = l + 1;
    }
    return left;
}

// Generate Gaussian-labeled dataset
vector<Point> generate_dataset(int l, double p, double q, mt19937& gen) {
    normal_distribution<double> dist(0.0, 1.0);
    uniform_real_distribution<double> urand(0.0, 1.0);

    vector<Point> dataset;
    for (int i = 0; i < l; ++i) {
        double x1 = dist(gen);
        double x2 = dist(gen);
        int label = (x1 >= 0) ? (urand(gen) < p ? 0 : 1) : (urand(gen) < q ? 0 : 1);
        dataset.push_back({x1, x2, label});
    }
    return dataset;
}

// Brute-force search for strategy with minimal empirical risk
void find_best_strategy(const vector<Point>& data, vector<double>& best_a, double& best_theta) {
    double min_risk = numeric_limits<double>::max();
    double rate = 0.2;
    auto a1 = {-1.0, 1.0}, a2 = {-1.0, 1.0}, theta = {-2.0, 2.0};
    auto steps  = [](double init, double target, double rate){ return (static_cast<int>((target - init)/rate));};

    for (int idx_a1 = 0, steps_a1 = steps(*a1.begin(),*(a1.begin()+1), rate); idx_a1 <= steps_a1; ++idx_a1) {
        double a1_ = (*a1.begin()) + idx_a1 * rate;

        for (int idx_a2 = 0, steps_a2 = steps(*a2.begin(),*(a2.begin()+1), rate); idx_a2 <= steps_a2; ++idx_a2) {
            double a2_ = (*a2.begin()) + idx_a2 * rate;

            for (int idx_theta = 0, steps_theta = steps(*theta.begin(),*(theta.begin()+1), rate); idx_theta <= steps_theta; ++idx_theta) {
                double theta_ = (*theta.begin()) + idx_theta * rate;

                vector<double> a = {a1_, a2_};
                double risk = empirical_risk(data, a, theta_);
                if (risk < min_risk) {
                    min_risk = risk;
                    best_a = a;
                    best_theta = theta_;
                }
            }
        }
    }
}

void save_to_csv(const string& filename, const vector<Point>& data) {
    ofstream file(filename);
    file << "x1,x2,label\n";
    for (const auto& pt : data)
        file << pt.x1 << "," << pt.x2 << "," << pt.label << "\n";
    file.close();
}

void save_parameters(const string& filename, const vector<double>& a, double theta) {
    ofstream file(filename);
    file << fixed << setprecision(6);
    file << "a1," << a[0] << "\n";
    file << "a2," << a[1] << "\n";
    file << "theta," << theta << "\n";
    file.close();
}

void append_summary_csv(const string& filename, double p, double q, double eps, double eta, int l_vc, double train_risk, double test_risk, double runtime_ms) {
    ofstream file(filename, ios::app);
    file << fixed << setprecision(6);
    file << p << "," << q << "," << eps << "," << eta << "," << l_vc << ","
         << train_risk << "," << test_risk << "," << runtime_ms << "\n";
    file.close();
}

void export_bounds_vs_epsilon(double eta, int vc_dim) {
    ofstream out("epsilon_vs_bounds.csv");
    out << "epsilon,l_simp,l_vc\n";
    double eps = 0.01, target = 0.1, rate = 0.005;
    int steps = static_cast<int>((target - eps) / rate);
    for (int idx_eps = 1; idx_eps <= steps; ++idx_eps)
    {
        eps += rate;
        int l_simp = compute_simplified_sample_size(eps, eta);
        int l_vc = compute_vc_sample_size(vc_dim, eps, eta);
        out << fixed << setprecision(3) << eps << "," << l_simp << "," << l_vc << "\n";
    }
    out.close();
    cout << "Exported epsilon_vs_bounds.csv for η = " << eta << "\n";
}

void run_experiment(double p, double q, double eta, double epsilon, int N, mt19937& gen, int vc_dim, const string& summary_path) {
    auto start = high_resolution_clock::now();

    int l_vc = compute_vc_sample_size(vc_dim, epsilon, eta);
    vector<Point> train_set = generate_dataset(l_vc, p, q, gen);
    vector<Point> test_set = generate_dataset(N, p, q, gen);
    export_bounds_vs_epsilon(eta, vc_dim);  // match η used in experiments


    vector<double> best_a;
    double best_theta;
    find_best_strategy(train_set, best_a, best_theta);

    double train_risk = empirical_risk(train_set, best_a, best_theta);
    double test_risk = empirical_risk(test_set, best_a, best_theta);

    auto end = high_resolution_clock::now();
    double elapsed_ms = duration<double, milli>(end - start).count();

    cout << "p=" << p << ", q=" << q << ", η=" << eta << ", ε=" << epsilon
         << ", l_vc=" << l_vc << ", train_risk=" << train_risk
         << ", test_risk=" << test_risk << ", runtime_ms=" << elapsed_ms << endl;

    string tag = "p" + to_string(int(p * 10)) + "_q" + to_string(int(q * 10)) +
                 "_e" + to_string(int(epsilon * 100)) + "_h" + to_string(int(eta * 100));

    save_to_csv(tag + "_train.csv", train_set);
    save_to_csv(tag + "_test.csv", test_set);
    save_parameters(tag + "_params.csv", best_a, best_theta);
    append_summary_csv(summary_path, p, q, epsilon, eta, l_vc, train_risk, test_risk, elapsed_ms);
}


int main() {
    random_device rd;
    mt19937 gen(rd());

    // Task parameters
    double p = 0.9, q = 0.1;
    double eta = 0.01, epsilon = 0.05;
    int N = 1000;

    // 1. VC-dimension
    int d_vc = compute_vc_dimension();
    cout << "1. VC-dimension (shattering test): " << d_vc << "\n";

    // 2. Required sample size from generalization bound
    int l = compute_simplified_sample_size(epsilon, eta);
    cout << "2. Required training sample size l (ε=" << epsilon << ", η=" << eta << "): " << l << "\n";

    // 3. Generate training set, find strategy, evaluate
    vector<Point> train_set = generate_dataset(l, p, q, gen);
    vector<double> best_a;
    double best_theta;
    find_best_strategy(train_set, best_a, best_theta);

    cout << "3. Best strategy parameters (a1, a2) = (" << best_a[0] << ", " << best_a[1] << ")\n";
    cout << "   Best threshold θ = " << best_theta << "\n";
    cout << "   Empirical risk on training set: " << empirical_risk(train_set, best_a, best_theta) << "\n";

    // 4. Test on new data
    vector<Point> test_set = generate_dataset(N, p, q, gen);
    double test_risk = empirical_risk(test_set, best_a, best_theta);
    cout << "4a. Risk on test set (theoretical risk approximation): " << test_risk << "\n";

    vector<Point> test_set_1 = generate_dataset(N, 0.9 * p , q , gen);
    double test_risk_1 = empirical_risk(test_set_1, best_a, best_theta);
    cout << "4b. Risk on test set_1 (theoretical risk approximation): " << test_risk_1 << "\n";
/**********************************************************************************************/
    vector<double> p_values = {0.9, 0.7};
    vector<double> q_values = {0.1, 0.3};
    vector<double> eps_values = {0.05, 0.1};
    vector<double> eta_values = {0.01, 0.05};

    const string summary_csv = "experiment_summary.csv";
    ofstream(summary_csv) << "p,q,epsilon,eta,l_vc,train_risk,test_risk,runtime_ms\n";

    for (double p : p_values)
        for (double q : q_values)
            for (double eps : eps_values)
                for (double eta : eta_values)
                    run_experiment(p, q, eta, eps, N, gen, d_vc, summary_csv);
    //export_bounds_vs_epsilon(eta, d_vc);  // match η used in experiments
    //system("venv/bin/python csv_plotting.py");



    return 0;
}

