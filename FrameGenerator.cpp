#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/frame.hpp>
#include <dv-sdk/processing/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <dv-sdk/processing/event.hpp>

#include <vector>
#include <cmath>

int64_t ONE_SECOND = 1e6; // microseconds

/**
 * An average time surface, as described by TODO: <Link>
 */
class AverageTimeSurface : public dv::TimeSurfaceBase<dv::EventStore>
{
public:
	int16_t R = 8; // Neighborhood size.
	int16_t halfR = 4;

	int16_t K = 8; // Cell size.
	int16_t cellWidth;
	int16_t cellHeight;
	int16_t nCells;
	cv::Mat cellLookup;
	std::vector<std::vector<dv::EventStore>> cellMemory; // Event store for each cell and polarity.

	int64_t tempWindow = 0.1 * ONE_SECOND; // Temporal Window (microseconds).
	double tau = 0.5;					   // Decay constant.

	cv::Mat timeSurface; // Holds the local time surface.
	cv::Size neighborhood;
	std::vector<std::vector<cv::Mat>> histograms; // Event store for each cell and polarity.

	cv::Mat ones;

	// Constructs a new, empty TimeSurface without any data allocated to it.
	AverageTimeSurface() = default;

	// Creates a new AverageTimeSurface of the given size.
	explicit AverageTimeSurface(const cv::Size &shape) : dv::TimeSurfaceBase<dv::EventStore>(shape)
	{
		cellWidth = shape.width / K;
		cellHeight = shape.height / K;
		nCells = cellHeight * cellWidth;

		cellLookup = cv::Mat::zeros(shape, CV_8U);
		neighborhood = cv::Size(2 * R + 1, 2 * R + 1);
		ones = cv::Mat::ones(neighborhood, CV_8U);

		// Initialize the cell lookup table.
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

		timeSurface = localTimeSurface(event, cellMemory.at(cell).at(polarityIndex));

		cv::Mat histogram;
		histograms.at(cell).at(polarityIndex).copyTo(histogram);

		cv::add(histogram, timeSurface, histograms.at(cell).at(polarityIndex));
	}

	cv::Mat localTimeSurface(dv::Event event_i, dv::EventStore memory)
	{
		cv::Mat localTimeSurface = cv::Mat::zeros(neighborhood, CV_8U);

		for (const dv::Event &event_j : memory)
		{
			double delta = (event_i.timestamp() - event_j.timestamp()) / ONE_SECOND;
			uchar value = std::exp(-delta / tau);

			int16_t shifted_y = event_j.y() - event_i.y() - R;
			int16_t shifted_x = event_j.x() - event_i.x() - R;

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
			cv::Mat onHistogram = cv::Mat::zeros(neighborhood, CV_8U);
			cv::Mat offHistogram = cv::Mat::zeros(neighborhood, CV_8U);

			std::vector<cv::Mat> histogramCell;

			histogramCell.push_back(onHistogram);
			histogramCell.push_back(offHistogram);

			histograms.push_back(histogramCell);
		}
	}
};

class FrameGenerator : public dv::ModuleBase
{
private:
	cv::Size inputSize;
	cv::Mat outFrame;
	AverageTimeSurface averageTimeSurface;

public:
	static void
	initInputs(dv::InputDefinitionList &in)
	{
		in.addEventInput("events");
	}

	static void initOutputs(dv::OutputDefinitionList &out)
	{
		out.addFrameOutput("frames");
	}

	static const char *initDescription()
	{
		return ("This module renders all events to frames");
	}

	static void initConfigOptions(dv::RuntimeConfig &config)
	{
		config.add("R", dv::ConfigOption::intOption("R", 32, 0, 32));
		config.add("Lookback", dv::ConfigOption::intOption("Time to look back", 1, 0, 255));
		config.setPriorityOptions({"R"});
	}

	FrameGenerator()
	{
		outputs.getFrameOutput("frames").setup(inputs.getEventInput("events"));
		inputSize = inputs.getEventInput("events").size();
		averageTimeSurface = AverageTimeSurface(inputSize);
	}

	void configUpdate() override
	{
	}

	void run() override
	{
		averageTimeSurface.accept(inputs.getEventInput("events").events());

		outputs.getFrameOutput("frames") << averageTimeSurface.histograms.at(120).at(1) << dv::commit;
	};
};

registerModuleClass(FrameGenerator)