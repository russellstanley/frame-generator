#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/frame.hpp>
#include <dv-sdk/processing/core.hpp>
#include <dv-sdk/processing/event.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>

int64_t ONE_SECOND = 1e6; // microseconds

/**
 * An average time surface, as described by TODO: <Link>
 */
class HistogramAverageTimeSurface : public dv::TimeSurfaceBase<dv::EventStore>
{
public:
	int16_t R = 8; // Neighborhood size.
	cv::Size neighborhood;

	int16_t K = 8; // Cell size.
	int16_t cellWidth;
	int16_t cellHeight;
	int16_t nCells;
	cv::Mat cellLookup;
	std::vector<std::vector<dv::EventStore>> cellMemory; // Event store for each cell and polarity.

	int64_t tempWindow = 0.1 * ONE_SECOND; // Temporal Window (microseconds).
	double tau = 0.5;					   // Decay constant.

	int windowSize = 30;
	cv::Mat timeSurface;									   // Holds the current local time surface.
	std::vector<std::vector<std::vector<cv::Mat>>> histograms; // Storage of the relevant local time surfaces.
	std::vector<std::vector<cv::Mat>> hats;					   // Hisogram of average time surfaces.

	// Constructs a new, empty TimeSurface without any data allocated to it.
	HistogramAverageTimeSurface() = default;

	// Creates a new Histogram of Average Time Surface class with the given size.
	explicit HistogramAverageTimeSurface(const cv::Size &shape) : dv::TimeSurfaceBase<dv::EventStore>(shape)
	{
		neighborhood = cv::Size(2 * R + 1, 2 * R + 1);

		// Initialize the cell lookup table.
		cellWidth = shape.width / K;
		cellHeight = shape.height / K;
		nCells = cellHeight * cellWidth;

		cellLookup = cv::Mat::zeros(shape, CV_8U);

		for (int i = 0; i < shape.width; i++)
		{
			for (int j = 0; j < shape.height; j++)
			{
				uchar pixel_row = i / K;
				uchar pixel_col = j / K;

				cellLookup.at<uchar>(i, j) = (pixel_row * cellWidth + pixel_col);
			}
		}

		// Initialize the cell memory table and histograms.
		reset();
	}

	// Inserts the event store into the time surface.
	void accept(const dv::EventStore &store) override
	{
		for (const dv::Event &event : store)
		{
			accept(event);
		}
	}

	// Inserts the event into the time surface.
	void accept(const typename dv::EventStore::iterator::value_type &event) override
	{
		int polarityIndex = 0;
		int cell = cellLookup.at<uchar>(event.y(), event.x());

		if (event.polarity())
		{
			polarityIndex = 1;
		}
		else
		{
			polarityIndex = 0;
		}

		// Add the new event to memory.
		cellMemory.at(cell).at(polarityIndex).push_back(event);

		// Filter the memory to remove events outside the temporal window.
		cellMemory.at(cell).at(polarityIndex) = filterMemory(cellMemory.at(cell).at(polarityIndex), event.timestamp());

		// Calulate the time surface for the given event.
		timeSurface = localTimeSurface(event, cellMemory.at(cell).at(polarityIndex));
		histograms.at(cell).at(polarityIndex).push_back(timeSurface);

		if (histograms.at(cell).at(polarityIndex).size() > windowSize)
		{
			cv::Mat subtract = histograms.at(cell).at(polarityIndex).at(0);
			histograms.at(cell).at(polarityIndex).erase(histograms.at(cell).at(polarityIndex).begin());

			// Subtract old time surface which is now outside the rolling window.
			cv::addWeighted(hats.at(cell).at(polarityIndex), 1.0, subtract, -1.0, 0.0, hats.at(cell).at(polarityIndex));
		}

		// Add the new time surface.
		cv::add(hats.at(cell).at(polarityIndex), timeSurface, hats.at(cell).at(polarityIndex));
	}

	cv::Mat localTimeSurface(dv::Event event_i, dv::EventStore memory)
	{
		cv::Mat localTimeSurface = cv::Mat::zeros(neighborhood, CV_8U);

		for (const dv::Event &event_j : memory)
		{
			double delta = (event_i.timestamp() - event_j.timestamp()) / ONE_SECOND;
			uchar value = std::exp(-delta / tau);

			int16_t shifted_y = event_j.y() - (event_i.y() - R);
			int16_t shifted_x = event_j.x() - (event_i.x() - R);

			localTimeSurface.at<uchar>(shifted_y, shifted_x) += value;
		}
		return localTimeSurface;
	}

	// Finds all events between the given time minus the length of the teporal window.
	dv::EventStore filterMemory(dv::EventStore memory, int64_t time)
	{
		int64_t timeLimit = memory.getHighestTime() - tempWindow;

		// Return all events which occur after the timestamp.
		return memory.sliceTime(timeLimit);
	}

	void reset()
	{
		cellMemory.clear();
		histograms.clear();

		// Initialize the cell event memory.
		for (int i = 0; i < nCells; i++)
		{
			// Create event storage for 'on' and 'off' events
			dv::EventStore onMemory;
			dv::EventStore offMemory;
			std::vector<dv::EventStore> memoryCell;

			memoryCell.push_back(onMemory);
			memoryCell.push_back(offMemory);

			cellMemory.push_back(memoryCell);
		}

		// Initialize the cell histogram storage.
		for (int i = 0; i < nCells; i++)
		{
			// Create a vector of images for 'on' and 'off' histograms.
			std::vector<cv::Mat> onHistogram;
			std::vector<cv::Mat> offHistogram;

			std::vector<std::vector<cv::Mat>> histogramCell;

			histogramCell.push_back(onHistogram);
			histogramCell.push_back(offHistogram);

			histograms.push_back(histogramCell);
		}

		// Initialize the HATS storage.
		for (int i = 0; i < nCells; i++)
		{
			// Create a vector of images for 'on' and 'off' histograms.
			cv::Mat onHistogram = cv::Mat::zeros(neighborhood, CV_8U);
			cv::Mat offHistogram = cv::Mat::zeros(neighborhood, CV_8U);

			std::vector<cv::Mat> hatsCell;

			hatsCell.push_back(onHistogram);
			hatsCell.push_back(offHistogram);

			hats.push_back(hatsCell);
		}
	}
};

class HatsGenerator : public dv::ModuleBase
{
private:
	cv::Mat outFrame;
	HistogramAverageTimeSurface hatsBase;

public:
	static void initInputs(dv::InputDefinitionList &in)
	{
		in.addEventInput("events");
	}

	static void initOutputs(dv::OutputDefinitionList &out)
	{
		out.addFrameOutput("frames");
	}

	static const char *initDescription()
	{
		return ("Renders incoming events as a Histogram of Average Time Surfaces");
	}

	static void initConfigOptions(dv::RuntimeConfig &config)
	{
		config.add("WindowSize", dv::ConfigOption::intOption("Window Size", 30, 5, 100));
		config.setPriorityOptions({"WindowSize"});
	}

	HatsGenerator()
	{
		cv::Size inputSize = inputs.getEventInput("events").size();
		hatsBase = HistogramAverageTimeSurface(inputSize);

		int sizeX = hatsBase.cellWidth * (2 * hatsBase.R + 1);
		int sizeY = hatsBase.cellHeight * (2 * hatsBase.R + 1);

		outputs.getFrameOutput("frames").setup(sizeX, sizeY, "description");
	}

	void configUpdate() override
	{
	}

	void run() override
	{
		hatsBase.accept(inputs.getEventInput("events").events());

		// Stack each histogram into a single frame.
		cv::Mat rows[16];

		for (int i = 0; i < 16; i++)
		{
			cv::Mat row[16];
			for (int j = 0; j < 16; j++)
			{
				row[j] = hatsBase.hats.at((16 * i) + j).at(1);
			}
			cv::hconcat(row, 16, rows[i]);
		}

		cv::Mat outFrame;
		cv::vconcat(rows, 16, outFrame);

		// Output the stacked frame.
		outputs.getFrameOutput("frames") << outFrame << dv::commit;
	};
};

registerModuleClass(HatsGenerator)