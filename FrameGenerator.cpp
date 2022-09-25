#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/frame.hpp>
#include <dv-sdk/processing/core.hpp>
#include <opencv2/imgproc.hpp>

#include <dv-sdk/processing/event.hpp>

/**
 * An average time surface, as described by TODO: <Link>
 */
class AverageTimeSurface : public dv::TimeSurfaceBase<dv::EventStore>
{
private:
	static constexpr int16_t DEFAULT_PATCH_WIDTH{10};
	int16_t mHalfPatchWidth;

public:
	// Constructs a new, empty TimeSurface without any data allocated to it.
	AverageTimeSurface() = default;

	// Creates a new AverageTimeSurface of the given size.
	explicit AverageTimeSurface(const cv::Size &shape) : dv::TimeSurfaceBase<dv::EventStore>(shape)
	{
		setPatchWidth(DEFAULT_PATCH_WIDTH);
	}

	// Set new patch width. The width of the patch surrounding arriving events which will be influenced by said events.
	void setPatchWidth(const int16_t patchWidth)
	{
		mHalfPatchWidth = patchWidth / 2;
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
		const auto rowStart = std::max(0, event.y() - mHalfPatchWidth);
		const auto rowEnd = std::min(AverageTimeSurface::rows() - 1, event.y() + mHalfPatchWidth);
		const auto colStart = std::max(0, event.x() - mHalfPatchWidth);
		const auto colEnd = std::min(AverageTimeSurface::cols() - 1, event.x() + mHalfPatchWidth);

		auto &currentPixel = AverageTimeSurface::at(event.y(), event.x());

		for (int16_t row = rowStart; row <= rowEnd; row++)
		{
			for (int16_t col = colStart; col <= colEnd; col++)
			{
				if (AverageTimeSurface::at(row, col) > currentPixel)
				{
					AverageTimeSurface::at(row, col) -= 1;
				}
			}
		}
		currentPixel = 255;
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
		cv::Mat outFrame = cv::Mat::zeros(inputSize, CV_8UC3);

		averageTimeSurface.accept(inputs.getEventInput("events").events());

		outFrame = averageTimeSurface.getOCVMatScaled();

		outputs.getFrameOutput("frames") << outFrame << dv::commit;
	};
};

registerModuleClass(FrameGenerator)