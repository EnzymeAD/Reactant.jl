using Test
using Reactant

using Dates
using Dates: value, UTInstant

const RDExt = Base.get_extension(Reactant, :ReactantDatesExt)

# Preparations for timestepper MWE unit test in the end
# Can't do that in the @testset scope, as scope issues occur
# Inspired by the usage in SpeedyWeather.jl

# Minimal clock-like mutable struct
mutable struct Clock{I,T,TS}
    n_timesteps::I
    time::T
    time_step::TS
end

# Minimal state
struct State{C}
    clock::C
end

function timestep!(state::State)
    state.clock.time += state.clock.time_step
    return nothing
end

function timestepping!(state::State)
    (; clock) = state

    @trace for i in 1:(clock.n_timesteps)
        timestep!(state)
    end

    return nothing
end

@testset "ReactantDatesExt" begin

    # --- Conversions: Dates ↔ Reactant roundtrips ---
    @testset "Period conversions: Dates → Reactant → Dates" begin
        for (DatesT, TracedT) in (
            (Dates.Year, RDExt.ReactantYear),
            (Dates.Quarter, RDExt.ReactantQuarter),
            (Dates.Month, RDExt.ReactantMonth),
            (Dates.Week, RDExt.ReactantWeek),
            (Dates.Day, RDExt.ReactantDay),
            (Dates.Hour, RDExt.ReactantHour),
            (Dates.Minute, RDExt.ReactantMinute),
            (Dates.Second, RDExt.ReactantSecond),
            (Dates.Millisecond, RDExt.ReactantMillisecond),
            (Dates.Microsecond, RDExt.ReactantMicrosecond),
            (Dates.Nanosecond, RDExt.ReactantNanosecond),
        )
            for v in (0, 42)
                orig = DatesT(v)
                traced = convert(TracedT, orig)
                @test traced isa TracedT
                @test value(traced) == v
                back = convert(DatesT, traced)
                @test back == orig
            end

            # Parametric convert: TracedT{Int64}
            orig = DatesT(5)
            traced_typed = convert(TracedT{Int64}, orig)
            @test traced_typed isa TracedT{Int64}
            @test value(traced_typed) == 5

            # Direct constructor
            @test value(TracedT(orig)) == value(orig)
        end
    end

    @testset "Cross-period conversions: Dates.Source → ReactantTarget" begin
        # All cross-period pairs that Dates.convert supports natively
        cross_pairs = (
            # DatePeriod → DatePeriod
            (Dates.Year, Dates.Quarter, RDExt.ReactantQuarter),
            (Dates.Year, Dates.Month, RDExt.ReactantMonth),
            (Dates.Quarter, Dates.Month, RDExt.ReactantMonth),
            (Dates.Week, Dates.Day, RDExt.ReactantDay),
            # DatePeriod → TimePeriod
            (Dates.Week, Dates.Hour, RDExt.ReactantHour),
            (Dates.Week, Dates.Minute, RDExt.ReactantMinute),
            (Dates.Week, Dates.Second, RDExt.ReactantSecond),
            (Dates.Week, Dates.Millisecond, RDExt.ReactantMillisecond),
            (Dates.Week, Dates.Microsecond, RDExt.ReactantMicrosecond),
            (Dates.Week, Dates.Nanosecond, RDExt.ReactantNanosecond),
            (Dates.Day, Dates.Hour, RDExt.ReactantHour),
            (Dates.Day, Dates.Minute, RDExt.ReactantMinute),
            (Dates.Day, Dates.Second, RDExt.ReactantSecond),
            (Dates.Day, Dates.Millisecond, RDExt.ReactantMillisecond),
            (Dates.Day, Dates.Microsecond, RDExt.ReactantMicrosecond),
            (Dates.Day, Dates.Nanosecond, RDExt.ReactantNanosecond),
            # TimePeriod → TimePeriod
            (Dates.Hour, Dates.Minute, RDExt.ReactantMinute),
            (Dates.Hour, Dates.Second, RDExt.ReactantSecond),
            (Dates.Hour, Dates.Millisecond, RDExt.ReactantMillisecond),
            (Dates.Hour, Dates.Microsecond, RDExt.ReactantMicrosecond),
            (Dates.Hour, Dates.Nanosecond, RDExt.ReactantNanosecond),
            (Dates.Minute, Dates.Second, RDExt.ReactantSecond),
            (Dates.Minute, Dates.Millisecond, RDExt.ReactantMillisecond),
            (Dates.Minute, Dates.Microsecond, RDExt.ReactantMicrosecond),
            (Dates.Minute, Dates.Nanosecond, RDExt.ReactantNanosecond),
            (Dates.Second, Dates.Millisecond, RDExt.ReactantMillisecond),
            (Dates.Second, Dates.Microsecond, RDExt.ReactantMicrosecond),
            (Dates.Second, Dates.Nanosecond, RDExt.ReactantNanosecond),
            (Dates.Millisecond, Dates.Microsecond, RDExt.ReactantMicrosecond),
            (Dates.Millisecond, Dates.Nanosecond, RDExt.ReactantNanosecond),
            (Dates.Microsecond, Dates.Nanosecond, RDExt.ReactantNanosecond),
        )
        for (SrcT, DstT, TracedT) in cross_pairs
            src = SrcT(1)
            expected = convert(DstT, src)
            traced = convert(TracedT, src)
            @test traced isa TracedT
            @test value(traced) == value(expected)

            # Parametric variant
            traced_p = convert(TracedT{Int64}, src)
            @test traced_p isa TracedT{Int64}
            @test value(traced_p) == value(expected)
        end
    end

    @testset "DateTime conversion: DateTime → ReactantDateTime → DateTime" begin
        dt = DateTime(2000, 1, 1)
        traced = convert(RDExt.ReactantDateTime, dt)
        @test traced isa RDExt.ReactantDateTime
        @test value(traced) == value(dt)
        back = convert(DateTime, traced)
        @test back == dt

        # Direct constructors
        dt = DateTime(2002, 6, 15)
        @test value(RDExt.ReactantDateTime(dt)) == value(dt)
        @test value(RDExt.ReactantDateTime{Int64}(dt)) == value(dt)
    end

    @testset "Date conversion: Date → ReactantDate → Date" begin
        d = Date(2000, 1, 1)
        traced = convert(RDExt.ReactantDate, d)
        @test traced isa RDExt.ReactantDate
        @test value(traced) == value(d)
        back = convert(Date, traced)
        @test back == d
    end

    @testset "Time conversion: Time → ReactantTime → Time" begin
        t = Time(12, 30, 45, 500, 250, 100)
        traced = convert(RDExt.ReactantTime, t)
        @test traced isa RDExt.ReactantTime
        @test value(traced) == value(t)
        back = convert(Time, traced)
        @test back == t
    end

    @testset "Time conversion: Time → ReactantTime → Time" begin
        t = Time(12, 30)
        @test value(RDExt.ReactantTime(t)) == value(t)
        @test value(RDExt.ReactantTime{Int64}(t)) == value(t)
    end

    # --- Accessors via @jit ---
    @testset "year, month, day accessors" begin
        dt = Dates.DateTime(2013, 7, 15)
        @test @jit(Dates.year(dt)) == Dates.year(dt)
        @test @jit(Dates.month(dt)) == Dates.month(dt)
        @test @jit(Dates.day(dt)) == Dates.day(dt)
        @test @jit(Dates.dayofmonth(dt)) == Dates.dayofmonth(dt)
        @test @jit(Dates.yearmonthday(dt)) == Dates.yearmonthday(dt)
        @test @jit(Dates.yearmonth(dt)) == Dates.yearmonth(dt)
        @test @jit(Dates.monthday(dt)) == Dates.monthday(dt)

        dt_r = convert(RDExt.ReactantDateTime, dt)
        @test @jit(Dates.year(dt_r)) == Dates.year(dt)
        @test @jit(Dates.month(dt_r)) == Dates.month(dt)
        @test @jit(Dates.day(dt_r)) == Dates.day(dt)
        @test @jit(Dates.dayofmonth(dt_r)) == Dates.dayofmonth(dt)
        @test @jit(Dates.yearmonthday(dt_r)) == Dates.yearmonthday(dt)
        @test @jit(Dates.yearmonth(dt_r)) == Dates.yearmonth(dt)
        @test @jit(Dates.monthday(dt_r)) == Dates.monthday(dt)

        d = Dates.Date(2013, 7, 15)
        @test @jit(Dates.year(d)) == Dates.year(d)
        @test @jit(Dates.month(d)) == Dates.month(d)
        @test @jit(Dates.day(d)) == Dates.day(d)
        @test @jit(Dates.dayofmonth(d)) == Dates.dayofmonth(d)
        @test @jit(Dates.yearmonthday(d)) == Dates.yearmonthday(d)
        @test @jit(Dates.yearmonth(d)) == Dates.yearmonth(d)
        @test @jit(Dates.monthday(d)) == Dates.monthday(d)

        d_r = convert(RDExt.ReactantDate, d)
        @test @jit(Dates.year(d_r)) == Dates.year(d)
        @test @jit(Dates.month(d_r)) == Dates.month(d)
        @test @jit(Dates.day(d_r)) == Dates.day(d)
        @test @jit(Dates.dayofmonth(d_r)) == Dates.dayofmonth(d)
        @test @jit(Dates.yearmonthday(d_r)) == Dates.yearmonthday(d)
        @test @jit(Dates.yearmonth(d_r)) == Dates.yearmonth(d)
        @test @jit(Dates.monthday(d_r)) == Dates.monthday(d)
    end

    @testset "hour, minute, second, millisecond accessors (DateTime)" begin
        dt = Dates.DateTime(2013, 7, 15, 12, 30, 45, 500)
        @test @jit(Dates.hour(dt)) == Dates.hour(dt)
        @test @jit(Dates.minute(dt)) == Dates.minute(dt)
        @test @jit(Dates.second(dt)) == Dates.second(dt)
        @test @jit(Dates.millisecond(dt)) == Dates.millisecond(dt)

        dt_r = convert(RDExt.ReactantDateTime, dt)
        @test @jit(Dates.hour(dt_r)) == Dates.hour(dt)
        @test @jit(Dates.minute(dt_r)) == Dates.minute(dt)
        @test @jit(Dates.second(dt_r)) == Dates.second(dt)
        @test @jit(Dates.millisecond(dt_r)) == Dates.millisecond(dt)
    end

    @testset "Time accessors" begin
        t = Dates.Time(12, 30, 45, 500, 250, 100)
        @test @jit(Dates.hour(t)) == Dates.hour(t)
        @test @jit(Dates.minute(t)) == Dates.minute(t)
        @test @jit(Dates.second(t)) == Dates.second(t)
        @test @jit(Dates.millisecond(t)) == Dates.millisecond(t)
        @test @jit(Dates.microsecond(t)) == Dates.microsecond(t)
        @test @jit(Dates.nanosecond(t)) == Dates.nanosecond(t)

        t_r = convert(RDExt.ReactantTime, t)
        @test @jit(Dates.hour(t_r)) == Dates.hour(t)
        @test @jit(Dates.minute(t_r)) == Dates.minute(t)
        @test @jit(Dates.second(t_r)) == Dates.second(t)
        @test @jit(Dates.millisecond(t_r)) == Dates.millisecond(t)
        @test @jit(Dates.microsecond(t_r)) == Dates.microsecond(t)
        @test @jit(Dates.nanosecond(t_r)) == Dates.nanosecond(t)
    end

    @testset "quarter accessor" begin
        d = Dates.Date(2000, 1, 1)
        @test @jit(Dates.quarter(d)) == Dates.quarter(d)

        d_r = convert(RDExt.ReactantDate, d)
        @test @jit(Dates.quarter(d_r)) == Dates.quarter(d)
    end

    @testset "week accessor" begin
        d = Dates.Date(2005, 1, 1)
        @test @jit(Dates.week(d)) == Dates.week(d)

        d_r = convert(RDExt.ReactantDate, d)
        @test @jit(Dates.week(d_r)) == Dates.week(d)
    end

    @testset "second, millisecond over various datetimes" begin
        for s in [0, 30, 59], ms in [0, 1, 500, 999]
            dt = Dates.DateTime(2013, 6, 15, 12, 0, s, ms)
            @test @jit(Dates.second(dt)) == s
            @test @jit(Dates.millisecond(dt)) == ms

            dt_r = convert(RDExt.ReactantDateTime, dt)
            @test @jit(Dates.second(dt_r)) == s
            @test @jit(Dates.millisecond(dt_r)) == ms
        end
    end

    # --- Arithmetic via @jit ---
    @testset "DateTime-Year arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        dt = convert(RDExt.ReactantDateTime, dt)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantYear(1))) ==
            Dates.DateTime(2000, 12, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantYear(100))) ==
            Dates.DateTime(2099, 12, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantYear(1000))) ==
            Dates.DateTime(2999, 12, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantYear(1))) ==
            Dates.DateTime(1998, 12, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantYear(100))) ==
            Dates.DateTime(1899, 12, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantYear(1000))) ==
            Dates.DateTime(999, 12, 27)

        # Leap year edge case: Feb 29 -> non-leap year clamps to Feb 28
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(2000, 2, 29))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantYear(1))) ==
            Dates.DateTime(2001, 2, 28)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantYear(1))) ==
            Dates.DateTime(1999, 2, 28)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantYear(4))) ==
            Dates.DateTime(2004, 2, 29)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantYear(4))) ==
            Dates.DateTime(1996, 2, 29)

        # Preserves time-of-day
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1972, 6, 30, 23, 59, 59))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantYear(1))) ==
            Dates.DateTime(1973, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantYear(1))) ==
            Dates.DateTime(1971, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantYear(-1))) ==
            Dates.DateTime(1971, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantYear(-1))) ==
            Dates.DateTime(1973, 6, 30, 23, 59, 59)

        # Verify all components after Year add
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(2000, 1, 1, 12, 30, 45, 500))
        r = @jit(dt + RDExt.ReactantYear(1))
        @test @jit(Dates.year(dt + RDExt.ReactantYear(1))) == 2001
        @test @jit(Dates.month(dt + RDExt.ReactantYear(1))) == 1
        @test @jit(Dates.day(dt + RDExt.ReactantYear(1))) == 1
        @test @jit(Dates.hour(dt + RDExt.ReactantYear(1))) == 12
        @test @jit(Dates.minute(dt + RDExt.ReactantYear(1))) == 30
        @test @jit(Dates.second(dt + RDExt.ReactantYear(1))) == 45
        @test @jit(Dates.millisecond(dt + RDExt.ReactantYear(1))) == 500
    end

    @testset "Date-Year arithmetic" begin
        dt = convert(RDExt.ReactantDate, Dates.Date(1999, 12, 27))
        @test Dates.Date(@jit(dt + RDExt.ReactantYear(1))) == Dates.Date(2000, 12, 27)
        @test Dates.Date(@jit(dt + RDExt.ReactantYear(100))) == Dates.Date(2099, 12, 27)
        @test Dates.Date(@jit(dt + RDExt.ReactantYear(1000))) == Dates.Date(2999, 12, 27)
        @test Dates.Date(@jit(dt - RDExt.ReactantYear(1))) == Dates.Date(1998, 12, 27)
        @test Dates.Date(@jit(dt - RDExt.ReactantYear(100))) == Dates.Date(1899, 12, 27)
        @test Dates.Date(@jit(dt - RDExt.ReactantYear(1000))) == Dates.Date(999, 12, 27)

        dt = convert(RDExt.ReactantDate, Dates.Date(2000, 2, 29))
        @test Dates.Date(@jit(dt + RDExt.ReactantYear(1))) == Dates.Date(2001, 2, 28)
        @test Dates.Date(@jit(dt - RDExt.ReactantYear(1))) == Dates.Date(1999, 2, 28)
        @test Dates.Date(@jit(dt + RDExt.ReactantYear(4))) == Dates.Date(2004, 2, 29)
        @test Dates.Date(@jit(dt - RDExt.ReactantYear(4))) == Dates.Date(1996, 2, 29)
    end

    @testset "DateTime-Quarter arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantQuarter(1))) ==
            Dates.DateTime(2000, 3, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantQuarter(-1))) ==
            Dates.DateTime(1999, 9, 27)
    end

    @testset "Date-Quarter arithmetic" begin
        dt = convert(RDExt.ReactantDate, Dates.Date(1999, 12, 27))
        @test Dates.Date(@jit(dt + RDExt.ReactantQuarter(1))) == Dates.Date(2000, 3, 27)
        @test Dates.Date(@jit(dt - RDExt.ReactantQuarter(1))) == Dates.Date(1999, 9, 27)
    end

    @testset "DateTime-Month arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(1))) ==
            Dates.DateTime(2000, 1, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(-1))) ==
            Dates.DateTime(1999, 11, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(-11))) ==
            Dates.DateTime(1999, 1, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(11))) ==
            Dates.DateTime(2000, 11, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(-12))) ==
            Dates.DateTime(1998, 12, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(12))) ==
            Dates.DateTime(2000, 12, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(13))) ==
            Dates.DateTime(2001, 1, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(100))) ==
            Dates.DateTime(2008, 4, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(1000))) ==
            Dates.DateTime(2083, 4, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMonth(1))) ==
            Dates.DateTime(1999, 11, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMonth(-1))) ==
            Dates.DateTime(2000, 1, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMonth(100))) ==
            Dates.DateTime(1991, 8, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMonth(1000))) ==
            Dates.DateTime(1916, 8, 27)

        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(2000, 2, 29))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(1))) ==
            Dates.DateTime(2000, 3, 29)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMonth(1))) ==
            Dates.DateTime(2000, 1, 29)

        # Preserves time-of-day
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1972, 6, 30, 23, 59, 59))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(1))) ==
            Dates.DateTime(1972, 7, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMonth(1))) ==
            Dates.DateTime(1972, 5, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMonth(-1))) ==
            Dates.DateTime(1972, 5, 30, 23, 59, 59)
    end

    @testset "Date-Month arithmetic" begin
        dt = convert(RDExt.ReactantDate, Dates.Date(1999, 12, 27))
        @test Dates.Date(@jit(dt + RDExt.ReactantMonth(1))) == Dates.Date(2000, 1, 27)
        @test Dates.Date(@jit(dt + RDExt.ReactantMonth(100))) == Dates.Date(2008, 4, 27)
        @test Dates.Date(@jit(dt + RDExt.ReactantMonth(1000))) == Dates.Date(2083, 4, 27)
        @test Dates.Date(@jit(dt - RDExt.ReactantMonth(1))) == Dates.Date(1999, 11, 27)
        @test Dates.Date(@jit(dt - RDExt.ReactantMonth(100))) == Dates.Date(1991, 8, 27)
        @test Dates.Date(@jit(dt - RDExt.ReactantMonth(1000))) == Dates.Date(1916, 8, 27)

        dt = convert(RDExt.ReactantDate, Dates.Date(2000, 2, 29))
        @test Dates.Date(@jit(dt + RDExt.ReactantMonth(1))) == Dates.Date(2000, 3, 29)
        @test Dates.Date(@jit(dt - RDExt.ReactantMonth(1))) == Dates.Date(2000, 1, 29)
    end

    @testset "DateTime-Week arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantWeek(1))) == Dates.DateTime(2000, 1, 3)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantWeek(100))) ==
            Dates.DateTime(2001, 11, 26)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantWeek(1000))) ==
            Dates.DateTime(2019, 2, 25)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantWeek(1))) ==
            Dates.DateTime(1999, 12, 20)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantWeek(100))) ==
            Dates.DateTime(1998, 1, 26)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantWeek(1000))) ==
            Dates.DateTime(1980, 10, 27)

        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(2000, 2, 29))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantWeek(1))) == Dates.DateTime(2000, 3, 7)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantWeek(1))) ==
            Dates.DateTime(2000, 2, 22)

        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1972, 6, 30, 23, 59, 59))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantWeek(1))) ==
            Dates.DateTime(1972, 7, 7, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantWeek(1))) ==
            Dates.DateTime(1972, 6, 23, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantWeek(-1))) ==
            Dates.DateTime(1972, 6, 23, 23, 59, 59)
    end

    @testset "DateTime-Day arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantDay(1))) ==
            Dates.DateTime(1999, 12, 28)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantDay(100))) ==
            Dates.DateTime(2000, 4, 5)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantDay(1000))) ==
            Dates.DateTime(2002, 9, 22)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantDay(1))) ==
            Dates.DateTime(1999, 12, 26)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantDay(100))) ==
            Dates.DateTime(1999, 9, 18)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantDay(1000))) ==
            Dates.DateTime(1997, 4, 1)

        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1972, 6, 30, 23, 59, 59))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantDay(1))) ==
            Dates.DateTime(1972, 7, 1, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantDay(1))) ==
            Dates.DateTime(1972, 6, 29, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantDay(-1))) ==
            Dates.DateTime(1972, 6, 29, 23, 59, 59)
    end

    @testset "Date-Week arithmetic" begin
        dt = convert(RDExt.ReactantDate, Dates.Date(1999, 12, 27))
        @test Dates.Date(@jit(dt + RDExt.ReactantWeek(1))) == Dates.Date(2000, 1, 3)
        @test Dates.Date(@jit(dt + RDExt.ReactantWeek(100))) == Dates.Date(2001, 11, 26)
        @test Dates.Date(@jit(dt + RDExt.ReactantWeek(1000))) == Dates.Date(2019, 2, 25)
        @test Dates.Date(@jit(dt - RDExt.ReactantWeek(1))) == Dates.Date(1999, 12, 20)
        @test Dates.Date(@jit(dt - RDExt.ReactantWeek(100))) == Dates.Date(1998, 1, 26)
        @test Dates.Date(@jit(dt - RDExt.ReactantWeek(1000))) == Dates.Date(1980, 10, 27)

        dt = convert(RDExt.ReactantDate, Dates.Date(2000, 2, 29))
        @test Dates.Date(@jit(dt + RDExt.ReactantWeek(1))) == Dates.Date(2000, 3, 7)
        @test Dates.Date(@jit(dt - RDExt.ReactantWeek(1))) == Dates.Date(2000, 2, 22)
    end

    @testset "Date-Day arithmetic" begin
        dt = convert(RDExt.ReactantDate, Dates.Date(1999, 12, 27))
        @test Dates.Date(@jit(dt + RDExt.ReactantDay(1))) == Dates.Date(1999, 12, 28)
        @test Dates.Date(@jit(dt + RDExt.ReactantDay(100))) == Dates.Date(2000, 4, 5)
        @test Dates.Date(@jit(dt + RDExt.ReactantDay(1000))) == Dates.Date(2002, 9, 22)
        @test Dates.Date(@jit(dt - RDExt.ReactantDay(1))) == Dates.Date(1999, 12, 26)
        @test Dates.Date(@jit(dt - RDExt.ReactantDay(100))) == Dates.Date(1999, 9, 18)
        @test Dates.Date(@jit(dt - RDExt.ReactantDay(1000))) == Dates.Date(1997, 4, 1)
    end

    @testset "DateTime-Hour arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantHour(1))) ==
            Dates.DateTime(1999, 12, 27, 1)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantHour(100))) ==
            Dates.DateTime(1999, 12, 31, 4)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantHour(1000))) ==
            Dates.DateTime(2000, 2, 6, 16)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantHour(1))) ==
            Dates.DateTime(1999, 12, 26, 23)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantHour(100))) ==
            Dates.DateTime(1999, 12, 22, 20)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantHour(1000))) ==
            Dates.DateTime(1999, 11, 15, 8)

        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1972, 6, 30, 23, 59, 59))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantHour(1))) ==
            Dates.DateTime(1972, 7, 1, 0, 59, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantHour(1))) ==
            Dates.DateTime(1972, 6, 30, 22, 59, 59)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantHour(-1))) ==
            Dates.DateTime(1972, 6, 30, 22, 59, 59)
    end

    @testset "DateTime-Minute arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMinute(1))) ==
            Dates.DateTime(1999, 12, 27, 0, 1)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMinute(100))) ==
            Dates.DateTime(1999, 12, 27, 1, 40)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMinute(1000))) ==
            Dates.DateTime(1999, 12, 27, 16, 40)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMinute(1))) ==
            Dates.DateTime(1999, 12, 26, 23, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMinute(100))) ==
            Dates.DateTime(1999, 12, 26, 22, 20)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMinute(1000))) ==
            Dates.DateTime(1999, 12, 26, 7, 20)

        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1972, 6, 30, 23, 59, 59))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMinute(1))) ==
            Dates.DateTime(1972, 7, 1, 0, 0, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMinute(1))) ==
            Dates.DateTime(1972, 6, 30, 23, 58, 59)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMinute(-1))) ==
            Dates.DateTime(1972, 6, 30, 23, 58, 59)
    end

    @testset "DateTime-Second arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantSecond(1))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantSecond(100))) ==
            Dates.DateTime(1999, 12, 27, 0, 1, 40)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantSecond(1000))) ==
            Dates.DateTime(1999, 12, 27, 0, 16, 40)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantSecond(1))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantSecond(100))) ==
            Dates.DateTime(1999, 12, 26, 23, 58, 20)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantSecond(1000))) ==
            Dates.DateTime(1999, 12, 26, 23, 43, 20)
    end

    @testset "DateTime-Millisecond arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMillisecond(1))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMillisecond(100))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 100)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMillisecond(1000))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 1)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMillisecond(1))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMillisecond(100))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 900)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMillisecond(1000))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59)

        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1972, 6, 30, 23, 59, 59))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMillisecond(1))) ==
            Dates.DateTime(1972, 6, 30, 23, 59, 59, 1)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMillisecond(1))) ==
            Dates.DateTime(1972, 6, 30, 23, 59, 58, 999)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMillisecond(-1))) ==
            Dates.DateTime(1972, 6, 30, 23, 59, 58, 999)
    end

    @testset "DateTime-Microsecond arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMicrosecond(1))) ==
            Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMicrosecond(501))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantMicrosecond(1499))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMicrosecond(1))) ==
            Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMicrosecond(501))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantMicrosecond(1499))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
    end

    @testset "DateTime-Nanosecond arithmetic" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(1999, 12, 27))
        @test Dates.DateTime(@jit(dt + RDExt.ReactantNanosecond(1))) ==
            Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantNanosecond(500_001))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + RDExt.ReactantNanosecond(1_499_999))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantNanosecond(1))) ==
            Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantNanosecond(500_001))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
        @test Dates.DateTime(@jit(dt - RDExt.ReactantNanosecond(1_499_999))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
    end

    @testset "Time-TimePeriod arithmetic" begin
        t = convert(RDExt.ReactantTime, Dates.Time(0))
        @test convert(Dates.Time, @jit(t + RDExt.ReactantHour(1))) == Dates.Time(1)
        @test convert(Dates.Time, @jit(t - RDExt.ReactantHour(1))) == Dates.Time(23)
        @test convert(Dates.Time, @jit(t - RDExt.ReactantNanosecond(1))) ==
            Dates.Time(23, 59, 59, 999, 999, 999)
        @test convert(Dates.Time, @jit(t + RDExt.ReactantNanosecond(-1))) ==
            Dates.Time(23, 59, 59, 999, 999, 999)
        @test convert(Dates.Time, @jit(t + RDExt.ReactantHour(24))) == Dates.Time(0)
        @test convert(Dates.Time, @jit(t + RDExt.ReactantNanosecond(86400000000000))) ==
            Dates.Time(0)
        @test convert(Dates.Time, @jit(t - RDExt.ReactantNanosecond(86400000000000))) ==
            Dates.Time(0)
        @test convert(Dates.Time, @jit(t + RDExt.ReactantMinute(1))) == Dates.Time(0, 1)
        @test convert(Dates.Time, @jit(t + RDExt.ReactantSecond(1))) == Dates.Time(0, 0, 1)
        @test convert(Dates.Time, @jit(t + RDExt.ReactantMillisecond(1))) ==
            Dates.Time(0, 0, 0, 1)
        @test convert(Dates.Time, @jit(t + RDExt.ReactantMicrosecond(1))) ==
            Dates.Time(0, 0, 0, 0, 1)
    end

    @testset "Commutativity" begin
        dt = convert(RDExt.ReactantDateTime, Dates.DateTime(2000))
        d = convert(RDExt.ReactantDate, Dates.Date(2000))
        t = convert(RDExt.ReactantTime, Dates.Time(12))

        @test Dates.DateTime(@jit(RDExt.ReactantYear(1) + dt)) ==
            Dates.DateTime(@jit(dt + RDExt.ReactantYear(1)))
        @test Dates.DateTime(@jit(RDExt.ReactantMonth(1) + dt)) ==
            Dates.DateTime(@jit(dt + RDExt.ReactantMonth(1)))
        @test Dates.DateTime(@jit(RDExt.ReactantDay(1) + dt)) ==
            Dates.DateTime(@jit(dt + RDExt.ReactantDay(1)))
        @test Dates.DateTime(@jit(RDExt.ReactantHour(1) + dt)) ==
            Dates.DateTime(@jit(dt + RDExt.ReactantHour(1)))

        @test Dates.Date(@jit(RDExt.ReactantYear(1) + d)) ==
            Dates.Date(@jit(d + RDExt.ReactantYear(1)))
        @test Dates.Date(@jit(RDExt.ReactantMonth(1) + d)) ==
            Dates.Date(@jit(d + RDExt.ReactantMonth(1)))
        @test Dates.Date(@jit(RDExt.ReactantWeek(1) + d)) ==
            Dates.Date(@jit(d + RDExt.ReactantWeek(1)))
        @test Dates.Date(@jit(RDExt.ReactantDay(1) + d)) ==
            Dates.Date(@jit(d + RDExt.ReactantDay(1)))

        @test convert(Dates.Time, @jit(RDExt.ReactantHour(1) + t)) ==
            convert(Dates.Time, @jit(t + RDExt.ReactantHour(1)))
        @test convert(Dates.Time, @jit(RDExt.ReactantMinute(1) + t)) ==
            convert(Dates.Time, @jit(t + RDExt.ReactantMinute(1)))
        @test convert(Dates.Time, @jit(RDExt.ReactantSecond(1) + t)) ==
            convert(Dates.Time, @jit(t + RDExt.ReactantSecond(1)))
        @test convert(Dates.Time, @jit(RDExt.ReactantNanosecond(1) + t)) ==
            convert(Dates.Time, @jit(t + RDExt.ReactantNanosecond(1)))
    end

    @testset "Month arithmetic non-associativity" begin
        a = convert(RDExt.ReactantDate, Dates.Date(2012, 1, 29))
        @test Dates.Date(@jit((a + RDExt.ReactantDay(1)) + RDExt.ReactantMonth(1))) !=
            Dates.Date(@jit((a + RDExt.ReactantMonth(1)) + RDExt.ReactantDay(1)))
        a = convert(RDExt.ReactantDate, Dates.Date(2012, 1, 30))
        @test Dates.Date(@jit((a + RDExt.ReactantDay(1)) + RDExt.ReactantMonth(1))) !=
            Dates.Date(@jit((a + RDExt.ReactantMonth(1)) + RDExt.ReactantDay(1)))
        a = convert(RDExt.ReactantDate, Dates.Date(2012, 2, 29))
        @test Dates.Date(@jit((a + RDExt.ReactantDay(1)) + RDExt.ReactantMonth(1))) !=
            Dates.Date(@jit((a + RDExt.ReactantMonth(1)) + RDExt.ReactantDay(1)))
    end

    @testset "toms and tons" begin
        # toms
        @test @jit(Dates.toms(RDExt.ReactantMillisecond(1))) == 1
        @test @jit(Dates.toms(RDExt.ReactantSecond(1))) == 1000
        @test @jit(Dates.toms(RDExt.ReactantMinute(1))) == 60000
        @test @jit(Dates.toms(RDExt.ReactantHour(1))) == 3600000
        @test @jit(Dates.toms(RDExt.ReactantDay(1))) == 86400000
        @test @jit(Dates.toms(RDExt.ReactantWeek(1))) == 604800000

        # tons
        @test @jit(Dates.tons(RDExt.ReactantNanosecond(1))) == 1
        @test @jit(Dates.tons(RDExt.ReactantMicrosecond(1))) == 1000
        @test @jit(Dates.tons(RDExt.ReactantMillisecond(1))) == 1000000
        @test @jit(Dates.tons(RDExt.ReactantSecond(1))) == 1000000000
        @test @jit(Dates.tons(RDExt.ReactantMinute(1))) == 60000000000
        @test @jit(Dates.tons(RDExt.ReactantHour(1))) == 3600000000000

        # days
        @test @jit(Dates.days(RDExt.ReactantDay(1))) == 1
        @test @jit(Dates.days(RDExt.ReactantWeek(1))) == 7
        @test @jit(Dates.days(RDExt.ReactantWeek(2))) == 14
    end

    @testset "datetime2julian" begin
        @test @jit(
            Dates.datetime2julian(convert(RDExt.ReactantDateTime, Dates.DateTime(2000)))
        ) == Dates.datetime2julian(Dates.DateTime(2000))
        @test @jit(
            Dates.datetime2julian(
                convert(RDExt.ReactantDateTime, Dates.DateTime(1930, 12, 1, 1, 5, 1))
            )
        ) == Dates.datetime2julian(Dates.DateTime(1930, 12, 1, 1, 5, 1))
    end

    @testset "Minimal timestepper with Dates" begin
        clock = Clock(5, DateTime(2002, 1, 1), Dates.Day(1))
        state = State(clock)
        timestepping!(state)

        # TODO: the track_numbers here is necessary. That may or may not be a bug within trace (track_numbers option in trace gives wrong results), see https://github.com/EnzymeAD/Reactant.jl/issues/2561
        #clock_jit = Reactant.to_rarray(
        #    Clock(5, DateTime(2002, 1, 1), Dates.Day(1)); #track_numbers = true
        #)
        clock_jit = Clock(5, Dates.DateTime(2002, 1, 1), Dates.Day(1))
        clock_jit = Reactant.to_rarray(clock_jit, track_numbers=true)

        state_jit = State(clock_jit)
        @jit timestepping!(state_jit)

        @test DateTime(state_jit.clock.time) == state.clock.time
    end
end
