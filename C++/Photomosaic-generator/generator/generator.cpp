// /std:c++17 /openmp
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <boost/program_options.hpp>
#include <omp.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace po = boost::program_options;

static std::array<int, 3> hist_size;

std::vector<cv::Mat> load_images(std::string images_path, int n, int x_step, int y_step) {
    std::vector<cv::Mat> resized_images;
    for (auto const& entry : std::filesystem::directory_iterator{ images_path }) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        cv::resize(img, img, { x_step, y_step });
        resized_images.push_back(img);

        if (resized_images.size() >= n)
            break;
    }
    return resized_images;
}

cv::Mat calc_hist(const cv::Mat& image) {
    cv::Mat hist;
    static int channels[]{ 0, 1, 2 };
    static float range[]{ 0, 256 };
    static const float* ranges[] = { range, range, range };
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, hist_size.data(), ranges);
    //cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

void generate_serially(const cv::Mat& target, const std::vector<cv::Mat>& resized_images, int x_step, int y_step) {
    std::vector<cv::Mat> hists;
    hists.resize(resized_images.size());
    for (size_t i = 0; i < hists.size(); i++) {
        hists[i] = calc_hist(resized_images[i]);
    }

    for (int y = 0; y < target.rows; y += x_step) {
        int y_end = std::min(y + y_step, target.rows);
        int y_size = y_end - y;

        for (int x = 0; x < target.cols; x += x_step) {
            int x_end = std::min(x + x_step, target.cols);
            int x_size = x_end - x;

            cv::Mat target_region = target({ x, y, x_size, y_size });
            cv::Mat target_hist = calc_hist(target_region);

            size_t max_i;
            double max_sim = 0;
            for (size_t i = 0; i < hists.size(); i++) {
                double sim = cv::compareHist(target_hist, hists[i], cv::HISTCMP_CORREL);
                if (sim > max_sim)
                    max_i = i, max_sim = sim;
            }
            resized_images[max_i]({ 0, 0, x_size, y_size }).copyTo(target_region);
        }
    }
}

void generate(const cv::Mat& target, const std::vector<cv::Mat>& resized_images, int x_step, int y_step) {
    std::vector<cv::Mat> hists;
    hists.resize(resized_images.size());
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < hists.size(); i++) {
            hists[i] = calc_hist(resized_images[i]);
        }

#pragma omp for
        for (int y = 0; y < target.rows; y += x_step) {
            int y_end = std::min(y + y_step, target.rows);
            int y_size = y_end - y;

            for (int x = 0; x < target.cols; x += x_step) {
                int x_end = std::min(x + x_step, target.cols);
                int x_size = x_end - x;

                cv::Mat target_region = target({ x, y, x_size, y_size });
                cv::Mat target_hist = calc_hist(target_region);

                size_t max_i;
                double max_sim = 0;
                for (size_t i = 0; i < hists.size(); i++) {
                    double sim = cv::compareHist(target_hist, hists[i], cv::HISTCMP_CORREL);
                    if (sim > max_sim)
                        max_i = i, max_sim = sim;
                }
                resized_images[max_i]({ 0, 0, x_size, y_size }).copyTo(target_region);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    std::string images_path, target_path, output_path;
    int n;
    int x_n, y_n;
    int precision;

    // CLI
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("images,I", po::value<std::string>(&images_path), "* directory of the source images")
        (",n", po::value<int>(&n), "limit the maximum number of source images")
        ("target,t", po::value<std::string>(&target_path), "* target image")
        ("xn", po::value<int>(&x_n)->default_value(100), "number of x blocks")
        ("yn", po::value<int>(&y_n)->default_value(150), "number of y blocks")
        ("precision", po::value<int>(&precision)->default_value(8), "histogram size")
        ("threads",  po::value<int>(), "number of threads for generating Photomosaic (0 for serial)")
        ("output,o", po::value<std::string>(&output_path), "output path")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || !vm.count("images") || !vm.count("target")) {
        std::cout << desc << '\n';
        return 1;
    }

    if (!vm.count("n"))
        n = std::numeric_limits<int>::max();

    hist_size = { precision, precision, precision };


    // Prepare
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);

    std::streamsize timer_precision = -std::log10(omp_get_wtick());
    std::cout << "Timer precision: " << omp_get_wtick() << "s\n\n";

    cv::Mat target = cv::imread(target_path, cv::IMREAD_COLOR);
    
    int x_step = target.cols / x_n, y_step = target.rows / y_n;
    
    // Load source images
    std::cout << "Loading source images...\n";
    double t_load = omp_get_wtime();
    std::vector<cv::Mat> resized_images = load_images(images_path, n, x_step, y_step);
    t_load = omp_get_wtime() - t_load;
    std::cout << "Time for loading source images: " << std::setprecision(timer_precision) << t_load << "s\n\n";
    
    // Generate Photomosaic
    std::cout << "Generating Photomosaic...\n";
    double t_g = omp_get_wtime();
    if (vm.count("threads")) {
        int threads = vm["threads"].as<int>();
        if (threads == 0)
            generate_serially(target, resized_images, x_step, y_step);
        else {
            omp_set_dynamic(false);
            omp_set_num_threads(threads);
            generate(target, resized_images, x_step, y_step);
        }
    }
    else {
        generate(target, resized_images, x_step, y_step);
    }
    t_g = omp_get_wtime() - t_g;
    std::cout << "Time for generating Photomosaic: " << std::setprecision(timer_precision) << t_g << "s\n";


    // Output result
    if (vm.count("output")) {
        cv::imwrite(output_path, target);
    }
    else {
        cv::imshow("Photographic", target);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}
