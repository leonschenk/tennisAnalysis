#include <opencv2/opencv.hpp>

using namespace std;

enum class Derivative {
    GAUSSIAN,
    DX,
    DY,
    DXX,
    DYY,
    DXY
};

double gaussian(int x, int y, double sigma) {
    return 1.0/(2.0 * CV_PI * pow(sigma, 2.0)) * exp(-(pow(double(x), 2.0) + pow(double(y), 2.0))/(2.0*pow(sigma, 2.0)));
}

double firstOrderDerivative(int x, double sigma) {
    return -(double(x) / (pow(sigma, 2.0)));
}

double secondOrderDerivative(int x, double sigma) {
    return (pow(double(x), 2.0) - pow(sigma, 2.0)) / pow(sigma, 4.0);
}

cv::Mat1d gaussianKernel(double sigma, Derivative derivative) {
    int intSize = (int(3.0 * sigma));
    cv::Size size = cv::Size(intSize * 2 + 1, intSize * 2 + 1);
    cv::Mat1d kernel = cv::Mat1d::zeros(size);
    for (int x = -intSize; x < intSize; x++) {
        for (int y = -intSize; y < intSize; y++) {
            switch (derivative) {
                case Derivative::GAUSSIAN:
                    kernel.at<double>(y + intSize, x + intSize) = gaussian(x, y, sigma);
                    break;
                case Derivative::DX:
                    kernel.at<double>(y + intSize, x + intSize) = firstOrderDerivative(x, sigma) * gaussian(x, y, sigma);
                    break;
                case Derivative::DY:
                    kernel.at<double>(y + intSize, x + intSize) = firstOrderDerivative(y, sigma) * gaussian(x, y, sigma);
                    break;
                case Derivative::DXX:
                    kernel.at<double>(y + intSize, x + intSize) = secondOrderDerivative(x, sigma) * gaussian(x, y, sigma);
                    break;
                case Derivative::DYY:
                    kernel.at<double>(y + intSize, x + intSize) = secondOrderDerivative(y, sigma) * gaussian(x, y, sigma);
                    break;
                case Derivative::DXY:
                    kernel.at<double>(y + intSize, x + intSize) = firstOrderDerivative(x, sigma) * firstOrderDerivative(y, sigma) * gaussian(x, y, sigma);
                    break;
            }
        }
    }
    return kernel;
}

void colorPicker(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Mat3d* image = reinterpret_cast<cv::Mat3d*>(userdata);
        cv::Vec3d vec = image->at<cv::Vec3d>(cv::Point(x, y));

        double angleDeg = vec[1] * 360.0 / CV_2PI;

        std::cout << "Angle at (" << x << "," << y << ") is: " << angleDeg << " r0 is: " << vec[0] << " strength: " << vec[2] << std::endl;
    }
}

class PixelDetails {
public:
    PixelDetails(double r0, double angle, double strength) : angle(angle), strength(strength), r0(r0) {
    }

public:
    double angle;
    double strength;
    double r0;
};

cv::Vec3d matrixMultiply(cv::Vec3d vector, cv::Matx44d matrix) {
    cv::Matx41d vec = cv::Matx41d(vector[0], vector[1], vector[2], 1.0);
    cv::Matx41d res = matrix * vec;
    return cv::Vec3d(res(0), res(1), res(2)) / res(3);
}

cv::Vec3d normalize(cv::Vec3d vector) {
    double length = sqrt(vector.dot(vector));
    return vector / length;
}

cv::Matx44d lookatMatrix(cv::Vec3d eye, cv::Vec3d target, cv::Vec3d up) {
    cv::Vec3d zaxis = normalize(eye - target);
    cv::Vec3d xaxis = normalize(up.cross(zaxis));
    cv::Vec3d yaxis = normalize(zaxis.cross(xaxis));

    cv::Matx44d lookAtMatrix = cv::Matx44d::eye();
    lookAtMatrix(0, 0) = xaxis[0];
    lookAtMatrix(0, 1) = xaxis[1];
    lookAtMatrix(0, 2) = xaxis[2];
    lookAtMatrix(1, 0) = yaxis[0];
    lookAtMatrix(1, 1) = yaxis[1];
    lookAtMatrix(1, 2) = yaxis[2];
    lookAtMatrix(2, 0) = zaxis[0];
    lookAtMatrix(2, 1) = zaxis[1];
    lookAtMatrix(2, 2) = zaxis[2];
    lookAtMatrix(0, 3) = -xaxis.dot(eye);
    lookAtMatrix(1, 3) = -yaxis.dot(eye);
    lookAtMatrix(2, 3) = -zaxis.dot(eye);

    return lookAtMatrix;
}

cv::Matx44d getProjectionMatrix(double aspectRatio, double angleOfView, double near, double far) {
    double scale = 1.0 / tan(angleOfView * 0.5 * CV_PI / 180.0);
    cv::Matx44d matrix = cv::Matx44d::zeros();
    matrix(0, 0) = scale / aspectRatio;
    matrix(1, 1) = scale;
    matrix(2, 2) = far / (near - far);
    matrix(2, 3) = far * near / (near - far);
    matrix(3, 2) = -1.0;
    return matrix;
}

std::vector<std::pair<cv::Vec3d, cv::Vec3d>> tennisField = std::vector<std::pair<cv::Vec3d, cv::Vec3d>>();

cv::Point transform(cv::Vec3d vertex, cv::Matx44d cam, cv::Matx44d proj, cv::Size screen) {
    cv::Vec3d view = matrixMultiply(vertex, cam);
    std::cout << view[0] << "," << view[1] << "," << view[2] << std::endl;
    cv::Vec3d point = matrixMultiply(view, proj);
    std::cout << point[0] << "," << point[1] << "," << point[2] << std::endl;

    double x = (-point[0] + 1.0) * double(screen.width) * 0.5;
    double y = (-point[1] + 1.0) * double(screen.height) * 0.5;

    std::cout << "x is " << x << " and y is " << y << std::endl << std::endl;

    return cv::Point(x, y);
}

int main()
{
    cv::Mat img = cv::imread("field.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not read the image: " << "E:\\tennisAnalysis\\field.png" << std::endl;
        return 1;
    }
    cv::Mat img2;
    cv::Size new_size = cv::Size(img.cols/2, img.rows/2);
    cv::resize(img, img2, new_size);
    
    cv::Mat1d new_image = cv::Mat1d::zeros(img2.size());
    cv::Mat1d field_image = cv::Mat1d::zeros(img2.size());

    cv::Mat1d kernelDx = gaussianKernel(4.0, Derivative::DX);
    cv::Mat1d kernelDy = gaussianKernel(4.0, Derivative::DY);

    cv::Mat1d kernelDxx = gaussianKernel(4.0, Derivative::DXX);
    cv::Mat1d kernelDyy = gaussianKernel(4.0, Derivative::DYY);
    cv::Mat1d kernelDxy = gaussianKernel(4.0, Derivative::DXY);
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;

    cv::Vec3d l = cv::Vec3d(183.0 / 255.0, 183.0 / 255.0, 183.0 / 255.0);
    cv::Vec3d v = cv::Vec3d(103.0 / 255.0, 103.0 / 255.0, 70.0 / 255.0);

    double dlv = sqrt(pow(l[0] - v[0], 2.0) + pow(l[1] - v[1], 2.0) + pow(l[2] - v[2], 2.0));

    for (int y = 0; y < img2.rows; y++) {
        for (int x = 0; x < img2.cols; x++) {
            double b = img2.at<cv::Vec3b>(y, x)[0] * 1.0 / 255.0;
            double r = img2.at<cv::Vec3b>(y, x)[1] * 1.0 / 255.0;
            double g = img2.at<cv::Vec3b>(y, x)[2] * 1.0 / 255.0;

            double dl = sqrt(pow(r - l[0], 2.0) + pow(g - l[1], 2.0) + pow(b - l[2], 2.0));
            double dv = sqrt(pow(r - v[0], 2.0) + pow(g - v[1], 2.0) + pow(b - v[2], 2.0));

            new_image.at<double>(y, x) = sqrt(pow(((dlv - dl) * dv), 2.0)) / pow(dlv, 2.0);
            field_image.at<double>(y, x) = -sqrt(dv);
        }
    }
    cv::imshow("SDFSDFS", new_image);

    cv::Mat1d imageDx = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d imageDy = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d imageDxx = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d imageDyy = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d imageDxy = cv::Mat1d::zeros(new_image.size());
    cv::filter2D(new_image, imageDx, -1, kernelDx);
    cv::filter2D(new_image, imageDy, -1, kernelDy);
    cv::filter2D(new_image, imageDxx, -1, kernelDxx);
    cv::filter2D(new_image, imageDyy, -1, kernelDyy);
    cv::filter2D(new_image, imageDxy, -1, kernelDxy);
//    cv::dilate(field_image, field_image, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(15, 15)));
//    cv::filter2D(field_image, field_image, -1, gaussianKernel(10.0, Derivative::GAUSSIAN));
    cv::Mat1d hessianEig1 = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d hessianEig2 = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d hessianDir = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d hessianScore = cv::Mat1d::zeros(new_image.size());

    cv::Mat1d angles = cv::Mat1d::zeros(new_image.size());
    cv::Mat1d distances = cv::Mat1d::zeros(new_image.size());

    for (int y = 0; y < hessianEig1.rows; y++) {
        for (int x = 0; x < hessianEig1.cols; x++) {
            double dxx = imageDxx.at<double>(y, x);
            double dyy = imageDyy.at<double>(y, x);
            double dxy = imageDxy.at<double>(y, x);

            double D = pow(dxx, 2.0) + pow(dyy, 2.0) + (4.0 * pow(dxy, 2.0)) - (2.0 * dxx * dyy);
            if (D < 0.0) {
                cout << "D smaller < 0.0: " << D << endl;
            }
            D = D < 0.0 ? 0.0 : D;

            double lambda1 = (dxx + dyy + sqrt(D)) / 2.0;
            double lambda2 = (dxx + dyy - sqrt(D)) / 2.0;

            double score = -lambda2 - abs(lambda1);
            score = score < 0.0 ? 0.0 : score;

            hessianEig1.at<double>(y, x) = lambda1;
            hessianEig2.at<double>(y, x) = lambda2;
            hessianDir.at<double>(y, x) = abs(atan((lambda1 - dxx) / dxy));
            hessianScore.at<double>(y, x) = score;

            double vx = 1.0;
            double vy = (lambda1 - dxx) / dxy;

            double angle = atan(vy / vx);
            angles.at<double>(y, x) = angle;
        }
    }

    cv::normalize(hessianScore, hessianScore, 1.0, 0.0, cv::NORM_MINMAX);
    cv::imshow("LineScore", hessianScore);

    cv::Mat3f hsvData = cv::Mat3f::zeros(new_image.size());
    cv::Mat3d dataMat = cv::Mat3d::zeros(new_image.size());
    for (int x = 0; x < new_image.size().width; x++) {
        for (int y = 0; y < new_image.size().height; y++) {
            double angle = angles.at<double>(y, x);
            double score = hessianScore.at<double>(y, x);

            double tana = tan(angle);
            double cota = 1.0 / tana;
            double b = double(y) - (tana * double(x));

            double r0 = abs(b) / sqrt(pow(tana, 2.0) + 1.0);

            dataMat.at<cv::Vec3d>(y, x) = cv::Vec3d(r0, angle, score);
            hsvData.at<cv::Vec3f>(y, x) = cv::Vec3f(float((angle + (CV_PI / 2.0)) * 360.0 / CV_PI), 1.0f, float(score));
        }
    }
    std::vector<PixelDetails> pixelDetails = std::vector<PixelDetails>();
    for (int x = 0; x < new_image.size().width; x++) {
        for (int y = 0; y < new_image.size().height; y++) {
            cv::Vec3d pixelData = dataMat.at<cv::Vec3d>(y, x);
            pixelDetails.push_back(PixelDetails(pixelData[0], pixelData[1], pixelData[2]));
        }
    }
    std::sort(pixelDetails.begin(), pixelDetails.end(), [](const PixelDetails& a, const PixelDetails& b) {return a.angle < b.angle;});
    
    cv::Mat1d angleMat = cv::Mat1d::zeros(cv::Size(900, 1201));
    for (const PixelDetails& pixel : pixelDetails) {
        int cursor = int(((pixel.angle / CV_PI) + 0.5) * 900.0);
        int indR0 = int(pixel.r0);

        double& pv = angleMat.at<double>(indR0, cursor);
        pv += pixel.strength;
//        pv = log(exp(pv) + pixel.strength);
    }
    cv::copyMakeBorder(angleMat, angleMat, 0, 0, 30, 30, cv::BORDER_WRAP);
    cv::filter2D(angleMat, angleMat, -1, gaussianKernel(10.0, Derivative::GAUSSIAN), cv::Point(-1, -1), 0.0, cv::BORDER_ISOLATED);
    angleMat = angleMat(cv::Rect(30, 0, angleMat.size().width - 60, angleMat.size().height));
    
    cv::normalize(angleMat, angleMat, 1.0, 0.0, cv::NORM_MINMAX);
    cv::imshow("AngleMat: ", angleMat);

//    cv::normalize(field_image, field_image, 1.0, 0.0, cv::NORM_MINMAX);
//    cv::imshow("FieldImage: ", field_image);

//    cv::Ptr<cv::plot::Plot2d> plotje = cv::plot::Plot2d::create(angleMat);
//    cv::Mat plot;
//    plotje->render(plot);
//    imshow("Plot: ", plot);

    cv::Mat rgb;
    cv::cvtColor(hsvData, rgb, cv::ColorConversionCodes::COLOR_HSV2BGR);
    cv::imshow("HSV representatie van de hoek en de score", rgb);
    cv::setMouseCallback("HSV representatie van de hoek en de score", colorPicker, &dataMat);

    double depth = 23.77;
    double halfDepth = 23.77 / 2.0;
    double width = 10.97;
    double halfWidth = width / 2.0;
    double doubleWidth = 1.37;
    double serviceFieldDepth = 6.4;
    double midLineDepth = 0.15;

    cv::Vec3d hoek_lo = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d hoek_ro = cv::Vec3d(width, 0.0, 0.0);
    cv::Vec3d hoek_lb = cv::Vec3d(0.0, 0.0, depth);
    cv::Vec3d hoek_rb = cv::Vec3d(width, 0.0, depth);

    cv::Vec3d hoek_dlo = cv::Vec3d(doubleWidth, 0.0, 0.0);
    cv::Vec3d hoek_dro = cv::Vec3d(width - doubleWidth, 0.0, 0.0);
    cv::Vec3d hoek_dlb = cv::Vec3d(doubleWidth, 0.0, depth);
    cv::Vec3d hoek_drb = cv::Vec3d(width - doubleWidth, 0.0, depth);

    cv::Vec3d hoek_slo = cv::Vec3d(doubleWidth, 0.0, halfDepth - serviceFieldDepth);
    cv::Vec3d hoek_smo = cv::Vec3d(halfWidth, 0.0, halfDepth - serviceFieldDepth);
    cv::Vec3d hoek_sro = cv::Vec3d(width - doubleWidth, 0.0, halfDepth - serviceFieldDepth);
    cv::Vec3d hoek_slb = cv::Vec3d(doubleWidth, 0.0, halfDepth + serviceFieldDepth);
    cv::Vec3d hoek_smb = cv::Vec3d(halfWidth, 0.0, halfDepth + serviceFieldDepth);
    cv::Vec3d hoek_srb = cv::Vec3d(width - doubleWidth, 0.0, halfDepth + serviceFieldDepth);

    cv::Vec3d hoek_moo = cv::Vec3d(halfWidth, 0.0, 0.0);
    cv::Vec3d hoek_mob = cv::Vec3d(halfWidth, 0.0, midLineDepth);
    cv::Vec3d hoek_mbb = cv::Vec3d(halfWidth, 0.0, depth);
    cv::Vec3d hoek_mbo = cv::Vec3d(halfWidth, 0.0, depth - midLineDepth);

    tennisField.push_back(std::make_pair(hoek_lo, hoek_ro));
    tennisField.push_back(std::make_pair(hoek_lo, hoek_lb));
    tennisField.push_back(std::make_pair(hoek_lb, hoek_rb));
    tennisField.push_back(std::make_pair(hoek_ro, hoek_rb));

    tennisField.push_back(std::make_pair(hoek_dlo, hoek_dlb));
    tennisField.push_back(std::make_pair(hoek_dro, hoek_drb));

    tennisField.push_back(std::make_pair(hoek_slo, hoek_sro));
    tennisField.push_back(std::make_pair(hoek_slb, hoek_srb));

    tennisField.push_back(std::make_pair(hoek_smo, hoek_smb));

    tennisField.push_back(std::make_pair(hoek_moo, hoek_mob));
    tennisField.push_back(std::make_pair(hoek_mbo, hoek_mbb));

    cv::Mat1d values = cv::Mat1d::zeros(new_image.size() / 2);
    cv::Matx44d projM = getProjectionMatrix(values.size().aspectRatio(), 20.0, 0.01, 300.0);
    cv::Matx44d camM = lookatMatrix(cv::Vec3d(halfWidth, 5.0, -20.0), cv::Vec3d(halfWidth * 0.5, 0.0, halfDepth + serviceFieldDepth), cv::Vec3d(0.0, 1.0, 0.0));

    for (auto line : tennisField) {
        cv::Point2i xy1 = transform(line.first, camM, projM, values.size());
        cv::Point2i xy2 = transform(line.second, camM, projM, values.size());

        cv::line(values, xy1, xy2, cv::Scalar(0.5));
    }
    cv::normalize(values, values, 1.0, 0.0, cv::NORM_MINMAX);
    cv::imshow("MAT", values);
    
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    return 0;
}
