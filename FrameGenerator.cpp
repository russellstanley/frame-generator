#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/frame.hpp>
#include <dv-sdk/processing/core.hpp>
#include <opencv2/imgproc.hpp>

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

	int64_t tempWindow = 0.1 * ONE_SECOND; // Temporal Window (microseconds)
	int64_t tau = 0.5 * ONE_SECOND;		   // Decay constant

	int16_t K = 8; // Cell size.
	int16_t cellWidth;
	int16_t cellHeight;
	int16_t nCells;

	cv::Mat cellLookup;
	cv::Mat timeSurface;
	std::vector<std::vector<dv::EventStore>> cellMemory;

	// Constructs a new, empty TimeSurface without any data allocated to it.
	AverageTimeSurface() = default;

	// Creates a new AverageTimeSurface of the given size.
	explicit AverageTimeSurface(const cv::Size &shape) : dv::TimeSurfaceBase<dv::EventStore>(shape)
	{
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
		int cell = cellLookup.at<uchar>(event.y(), event.x());
		int polarityIndex;

		if (event.polarity())
		{
			polarityIndex = 1;
		}
		else
		{
			polarityIndex = 0;
		}
		cellMemory.at(cell).at(polarityIndex).push_back(event);

		// cellMemory.at(cell).at(polarityIndex) = filterMemory(cellMemory.at(cell).at(polarityIndex), event.timestamp());

		timeSurface = localTimeSurface(event, cellMemory.at(cell).at(polarityIndex));
	}

	cv::Mat localTimeSurface(dv::Event event_i, dv::EventStore memory)
	{
		cv::Size neighborhood(2 * R + 1, 2 * R + 1);
		cv::Mat localTimeSurface = cv::Mat::zeros(neighborhood, CV_8U);

		int64_t t_i = event_i.timestamp();

		for (const dv::Event &event_j : memory)
		{
			int64_t delta = t_i - event_j.timestamp();
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
		int64_t limit = time - tempWindow;
		bool found = false;

		int right = memory.size() - 1;
		int left = 0;

		int midpoint = 0;
		int position = 0;

		// Use binary search to find the position.
		while (left <= right && !found)
		{
			midpoint = (left + right) / 2;
			if (memory.at(midpoint).timestamp() == limit)
			{
				position = midpoint;
				found = true;
			}
			else
			{
				if (limit <= memory.at(midpoint).timestamp())
				{
					right = midpoint - 1;
				}
				else
				{
					left = midpoint + 1;
				}
			}
		}
		return memory.slice(position, memory.size() - position);
	}

	void reset()
	{
		cellMemory.clear();

		for (int i = 0; i < nCells; i++)
		{
			dv::EventStore on_memory;
			dv::EventStore off_memory;
			std::vector<dv::EventStore> cell;

			cell.push_back(on_memory);
			cell.push_back(off_memory);

			cellMemory.push_back(cell);
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
		// cv::Mat outFrame = cv::Mat::zeros(inputSize, CV_8UC3);

		averageTimeSurface.accept(inputs.getEventInput("events").events());

		// outFrame = averageTimeSurface.getOCVMatScaled();

		log.debug << averageTimeSurface.cellMemory.at(128).at(1).size() << dv::logEnd;
		outputs.getFrameOutput("frames") << averageTimeSurface.timeSurface << dv::commit;
	};
};

registerModuleClass(FrameGenerator)