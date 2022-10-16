#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/frame.hpp>
#include <dv-sdk/processing/core.hpp>
#include <opencv2/imgproc.hpp>

#include <dv-sdk/processing/event.hpp>

#include <vector>

/**
 * An average time surface, as described by TODO: <Link>
 */
class AverageTimeSurface : public dv::TimeSurfaceBase<dv::EventStore>
{
public:
	int16_t R = 32; // Neighborhood size.
	int16_t halfR = 16;

	float tempWindow = 0.1;
	float tau = 0.5;

	int16_t K = 32; // Cell size.
	int16_t cellWidth;
	int16_t cellHeight;
	int16_t nCells;

	cv::Mat cellLookup;
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

		outputs.getFrameOutput("frames") << averageTimeSurface.cellLookup << dv::commit;
	};
};

registerModuleClass(FrameGenerator)