using Test
using Dates
using Dates: value, UTInstant
using Reactant

@testset "ReactantDatesExt" begin

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

        d = Dates.Date(2013, 7, 15)
        @test @jit(Dates.year(d)) == Dates.year(d)
        @test @jit(Dates.month(d)) == Dates.month(d)
        @test @jit(Dates.day(d)) == Dates.day(d)
        @test @jit(Dates.dayofmonth(d)) == Dates.dayofmonth(d)
        @test @jit(Dates.yearmonthday(d)) == Dates.yearmonthday(d)
        @test @jit(Dates.yearmonth(d)) == Dates.yearmonth(d)
        @test @jit(Dates.monthday(d)) == Dates.monthday(d)
    end

    @testset "hour, minute, second, millisecond accessors (DateTime)" begin
        dt = Dates.DateTime(2013, 7, 15, 12, 30, 45, 500)
        @test @jit(Dates.hour(dt)) == Dates.hour(dt)
        @test @jit(Dates.minute(dt)) == Dates.minute(dt)
        @test @jit(Dates.second(dt)) == Dates.second(dt)
        @test @jit(Dates.millisecond(dt)) == Dates.millisecond(dt)
    end

    @testset "Time accessors" begin
        t = Dates.Time(12, 30, 45, 500, 250, 100)
        @test @jit(Dates.hour(t)) == Dates.hour(t)
        @test @jit(Dates.minute(t)) == Dates.minute(t)
        @test @jit(Dates.second(t)) == Dates.second(t)
        @test @jit(Dates.millisecond(t)) == Dates.millisecond(t)
        @test @jit(Dates.microsecond(t)) == Dates.microsecond(t)
        @test @jit(Dates.nanosecond(t)) == Dates.nanosecond(t)
    end

    @testset "quarter accessor" begin
        @test @jit(Dates.quarter(Dates.Date(2000, 1, 1))) == 1
        @test @jit(Dates.quarter(Dates.Date(2000, 3, 31))) == 1
        @test @jit(Dates.quarter(Dates.Date(2000, 4, 1))) == 2
        @test @jit(Dates.quarter(Dates.Date(2000, 6, 30))) == 2
        @test @jit(Dates.quarter(Dates.Date(2000, 7, 1))) == 3
        @test @jit(Dates.quarter(Dates.Date(2000, 9, 30))) == 3
        @test @jit(Dates.quarter(Dates.Date(2000, 10, 1))) == 4
        @test @jit(Dates.quarter(Dates.Date(2000, 12, 31))) == 4
    end

    @testset "week accessor" begin
        @test @jit(Dates.week(Dates.Date(2005, 1, 1))) == 53
        @test @jit(Dates.week(Dates.Date(2005, 1, 2))) == 53
        @test @jit(Dates.week(Dates.Date(2005, 12, 31))) == 52
        @test @jit(Dates.week(Dates.Date(2007, 1, 1))) == 1
        @test @jit(Dates.week(Dates.Date(2008, 12, 29))) == 1
        @test @jit(Dates.week(Dates.Date(2009, 12, 31))) == 53
        @test @jit(Dates.week(Dates.Date(2010, 1, 1))) == 53
    end

    @testset "second, millisecond over various datetimes" begin
        for s in [0, 30, 59], ms in [0, 1, 500, 999]
            dt = Dates.DateTime(2013, 6, 15, 12, 0, s, ms)
            @test @jit(Dates.second(dt)) == s
            @test @jit(Dates.millisecond(dt)) == ms
        end
    end

    # --- Arithmetic via @jit ---
    @testset "DateTime-Year arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Year(1))) == Dates.DateTime(2000, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Year(100))) == Dates.DateTime(2099, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Year(1000))) == Dates.DateTime(2999, 12, 27)
        @test Dates.DateTime(@jit(dt - Dates.Year(1))) == Dates.DateTime(1998, 12, 27)
        @test Dates.DateTime(@jit(dt - Dates.Year(100))) == Dates.DateTime(1899, 12, 27)
        @test Dates.DateTime(@jit(dt - Dates.Year(1000))) == Dates.DateTime(999, 12, 27)

        # Leap year edge case: Feb 29 -> non-leap year clamps to Feb 28
        dt = Dates.DateTime(2000, 2, 29)
        @test Dates.DateTime(@jit(dt + Dates.Year(1))) == Dates.DateTime(2001, 2, 28)
        @test Dates.DateTime(@jit(dt - Dates.Year(1))) == Dates.DateTime(1999, 2, 28)
        @test Dates.DateTime(@jit(dt + Dates.Year(4))) == Dates.DateTime(2004, 2, 29)
        @test Dates.DateTime(@jit(dt - Dates.Year(4))) == Dates.DateTime(1996, 2, 29)

        # Preserves time-of-day
        dt = Dates.DateTime(1972, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Year(1))) ==
            Dates.DateTime(1973, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - Dates.Year(1))) ==
            Dates.DateTime(1971, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Year(-1))) ==
            Dates.DateTime(1971, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - Dates.Year(-1))) ==
            Dates.DateTime(1973, 6, 30, 23, 59, 59)

        # Verify all components after Year add
        dt = Dates.DateTime(2000, 1, 1, 12, 30, 45, 500)
        r = @jit(dt + Dates.Year(1))
        @test @jit(Dates.year(dt + Dates.Year(1))) == 2001
        @test @jit(Dates.month(dt + Dates.Year(1))) == 1
        @test @jit(Dates.day(dt + Dates.Year(1))) == 1
        @test @jit(Dates.hour(dt + Dates.Year(1))) == 12
        @test @jit(Dates.minute(dt + Dates.Year(1))) == 30
        @test @jit(Dates.second(dt + Dates.Year(1))) == 45
        @test @jit(Dates.millisecond(dt + Dates.Year(1))) == 500
    end

    @testset "Date-Year arithmetic" begin
        dt = Dates.Date(1999, 12, 27)
        @test Dates.Date(@jit(dt + Dates.Year(1))) == Dates.Date(2000, 12, 27)
        @test Dates.Date(@jit(dt + Dates.Year(100))) == Dates.Date(2099, 12, 27)
        @test Dates.Date(@jit(dt + Dates.Year(1000))) == Dates.Date(2999, 12, 27)
        @test Dates.Date(@jit(dt - Dates.Year(1))) == Dates.Date(1998, 12, 27)
        @test Dates.Date(@jit(dt - Dates.Year(100))) == Dates.Date(1899, 12, 27)
        @test Dates.Date(@jit(dt - Dates.Year(1000))) == Dates.Date(999, 12, 27)

        dt = Dates.Date(2000, 2, 29)
        @test Dates.Date(@jit(dt + Dates.Year(1))) == Dates.Date(2001, 2, 28)
        @test Dates.Date(@jit(dt - Dates.Year(1))) == Dates.Date(1999, 2, 28)
        @test Dates.Date(@jit(dt + Dates.Year(4))) == Dates.Date(2004, 2, 29)
        @test Dates.Date(@jit(dt - Dates.Year(4))) == Dates.Date(1996, 2, 29)
    end

    @testset "DateTime-Quarter arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Quarter(1))) == Dates.DateTime(2000, 3, 27)
        @test Dates.DateTime(@jit(dt + Dates.Quarter(-1))) == Dates.DateTime(1999, 9, 27)
    end

    @testset "Date-Quarter arithmetic" begin
        dt = Dates.Date(1999, 12, 27)
        @test Dates.Date(@jit(dt + Dates.Quarter(1))) == Dates.Date(2000, 3, 27)
        @test Dates.Date(@jit(dt - Dates.Quarter(1))) == Dates.Date(1999, 9, 27)
    end

    @testset "DateTime-Month arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(1))) == Dates.DateTime(2000, 1, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(-1))) == Dates.DateTime(1999, 11, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(-11))) == Dates.DateTime(1999, 1, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(11))) == Dates.DateTime(2000, 11, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(-12))) == Dates.DateTime(1998, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(12))) == Dates.DateTime(2000, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(13))) == Dates.DateTime(2001, 1, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(100))) == Dates.DateTime(2008, 4, 27)
        @test Dates.DateTime(@jit(dt + Dates.Month(1000))) == Dates.DateTime(2083, 4, 27)
        @test Dates.DateTime(@jit(dt - Dates.Month(1))) == Dates.DateTime(1999, 11, 27)
        @test Dates.DateTime(@jit(dt - Dates.Month(-1))) == Dates.DateTime(2000, 1, 27)
        @test Dates.DateTime(@jit(dt - Dates.Month(100))) == Dates.DateTime(1991, 8, 27)
        @test Dates.DateTime(@jit(dt - Dates.Month(1000))) == Dates.DateTime(1916, 8, 27)

        dt = Dates.DateTime(2000, 2, 29)
        @test Dates.DateTime(@jit(dt + Dates.Month(1))) == Dates.DateTime(2000, 3, 29)
        @test Dates.DateTime(@jit(dt - Dates.Month(1))) == Dates.DateTime(2000, 1, 29)

        # Preserves time-of-day
        dt = Dates.DateTime(1972, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Month(1))) ==
            Dates.DateTime(1972, 7, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - Dates.Month(1))) ==
            Dates.DateTime(1972, 5, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Month(-1))) ==
            Dates.DateTime(1972, 5, 30, 23, 59, 59)
    end

    @testset "Date-Month arithmetic" begin
        dt = Dates.Date(1999, 12, 27)
        @test Dates.Date(@jit(dt + Dates.Month(1))) == Dates.Date(2000, 1, 27)
        @test Dates.Date(@jit(dt + Dates.Month(100))) == Dates.Date(2008, 4, 27)
        @test Dates.Date(@jit(dt + Dates.Month(1000))) == Dates.Date(2083, 4, 27)
        @test Dates.Date(@jit(dt - Dates.Month(1))) == Dates.Date(1999, 11, 27)
        @test Dates.Date(@jit(dt - Dates.Month(100))) == Dates.Date(1991, 8, 27)
        @test Dates.Date(@jit(dt - Dates.Month(1000))) == Dates.Date(1916, 8, 27)

        dt = Dates.Date(2000, 2, 29)
        @test Dates.Date(@jit(dt + Dates.Month(1))) == Dates.Date(2000, 3, 29)
        @test Dates.Date(@jit(dt - Dates.Month(1))) == Dates.Date(2000, 1, 29)
    end

    @testset "DateTime-Week arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Week(1))) == Dates.DateTime(2000, 1, 3)
        @test Dates.DateTime(@jit(dt + Dates.Week(100))) == Dates.DateTime(2001, 11, 26)
        @test Dates.DateTime(@jit(dt + Dates.Week(1000))) == Dates.DateTime(2019, 2, 25)
        @test Dates.DateTime(@jit(dt - Dates.Week(1))) == Dates.DateTime(1999, 12, 20)
        @test Dates.DateTime(@jit(dt - Dates.Week(100))) == Dates.DateTime(1998, 1, 26)
        @test Dates.DateTime(@jit(dt - Dates.Week(1000))) == Dates.DateTime(1980, 10, 27)

        dt = Dates.DateTime(2000, 2, 29)
        @test Dates.DateTime(@jit(dt + Dates.Week(1))) == Dates.DateTime(2000, 3, 7)
        @test Dates.DateTime(@jit(dt - Dates.Week(1))) == Dates.DateTime(2000, 2, 22)

        dt = Dates.DateTime(1972, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Week(1))) ==
            Dates.DateTime(1972, 7, 7, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - Dates.Week(1))) ==
            Dates.DateTime(1972, 6, 23, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Week(-1))) ==
            Dates.DateTime(1972, 6, 23, 23, 59, 59)
    end

    @testset "DateTime-Day arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Day(1))) == Dates.DateTime(1999, 12, 28)
        @test Dates.DateTime(@jit(dt + Dates.Day(100))) == Dates.DateTime(2000, 4, 5)
        @test Dates.DateTime(@jit(dt + Dates.Day(1000))) == Dates.DateTime(2002, 9, 22)
        @test Dates.DateTime(@jit(dt - Dates.Day(1))) == Dates.DateTime(1999, 12, 26)
        @test Dates.DateTime(@jit(dt - Dates.Day(100))) == Dates.DateTime(1999, 9, 18)
        @test Dates.DateTime(@jit(dt - Dates.Day(1000))) == Dates.DateTime(1997, 4, 1)

        dt = Dates.DateTime(1972, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Day(1))) ==
            Dates.DateTime(1972, 7, 1, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - Dates.Day(1))) ==
            Dates.DateTime(1972, 6, 29, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Day(-1))) ==
            Dates.DateTime(1972, 6, 29, 23, 59, 59)
    end

    @testset "Date-Week arithmetic" begin
        dt = Dates.Date(1999, 12, 27)
        @test Dates.Date(@jit(dt + Dates.Week(1))) == Dates.Date(2000, 1, 3)
        @test Dates.Date(@jit(dt + Dates.Week(100))) == Dates.Date(2001, 11, 26)
        @test Dates.Date(@jit(dt + Dates.Week(1000))) == Dates.Date(2019, 2, 25)
        @test Dates.Date(@jit(dt - Dates.Week(1))) == Dates.Date(1999, 12, 20)
        @test Dates.Date(@jit(dt - Dates.Week(100))) == Dates.Date(1998, 1, 26)
        @test Dates.Date(@jit(dt - Dates.Week(1000))) == Dates.Date(1980, 10, 27)

        dt = Dates.Date(2000, 2, 29)
        @test Dates.Date(@jit(dt + Dates.Week(1))) == Dates.Date(2000, 3, 7)
        @test Dates.Date(@jit(dt - Dates.Week(1))) == Dates.Date(2000, 2, 22)
    end

    @testset "Date-Day arithmetic" begin
        dt = Dates.Date(1999, 12, 27)
        @test Dates.Date(@jit(dt + Dates.Day(1))) == Dates.Date(1999, 12, 28)
        @test Dates.Date(@jit(dt + Dates.Day(100))) == Dates.Date(2000, 4, 5)
        @test Dates.Date(@jit(dt + Dates.Day(1000))) == Dates.Date(2002, 9, 22)
        @test Dates.Date(@jit(dt - Dates.Day(1))) == Dates.Date(1999, 12, 26)
        @test Dates.Date(@jit(dt - Dates.Day(100))) == Dates.Date(1999, 9, 18)
        @test Dates.Date(@jit(dt - Dates.Day(1000))) == Dates.Date(1997, 4, 1)
    end

    @testset "DateTime-Hour arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Hour(1))) == Dates.DateTime(1999, 12, 27, 1)
        @test Dates.DateTime(@jit(dt + Dates.Hour(100))) == Dates.DateTime(1999, 12, 31, 4)
        @test Dates.DateTime(@jit(dt + Dates.Hour(1000))) == Dates.DateTime(2000, 2, 6, 16)
        @test Dates.DateTime(@jit(dt - Dates.Hour(1))) == Dates.DateTime(1999, 12, 26, 23)
        @test Dates.DateTime(@jit(dt - Dates.Hour(100))) == Dates.DateTime(1999, 12, 22, 20)
        @test Dates.DateTime(@jit(dt - Dates.Hour(1000))) == Dates.DateTime(1999, 11, 15, 8)

        dt = Dates.DateTime(1972, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Hour(1))) ==
            Dates.DateTime(1972, 7, 1, 0, 59, 59)
        @test Dates.DateTime(@jit(dt - Dates.Hour(1))) ==
            Dates.DateTime(1972, 6, 30, 22, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Hour(-1))) ==
            Dates.DateTime(1972, 6, 30, 22, 59, 59)
    end

    @testset "DateTime-Minute arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Minute(1))) ==
            Dates.DateTime(1999, 12, 27, 0, 1)
        @test Dates.DateTime(@jit(dt + Dates.Minute(100))) ==
            Dates.DateTime(1999, 12, 27, 1, 40)
        @test Dates.DateTime(@jit(dt + Dates.Minute(1000))) ==
            Dates.DateTime(1999, 12, 27, 16, 40)
        @test Dates.DateTime(@jit(dt - Dates.Minute(1))) ==
            Dates.DateTime(1999, 12, 26, 23, 59)
        @test Dates.DateTime(@jit(dt - Dates.Minute(100))) ==
            Dates.DateTime(1999, 12, 26, 22, 20)
        @test Dates.DateTime(@jit(dt - Dates.Minute(1000))) ==
            Dates.DateTime(1999, 12, 26, 7, 20)

        dt = Dates.DateTime(1972, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Minute(1))) ==
            Dates.DateTime(1972, 7, 1, 0, 0, 59)
        @test Dates.DateTime(@jit(dt - Dates.Minute(1))) ==
            Dates.DateTime(1972, 6, 30, 23, 58, 59)
        @test Dates.DateTime(@jit(dt + Dates.Minute(-1))) ==
            Dates.DateTime(1972, 6, 30, 23, 58, 59)
    end

    @testset "DateTime-Second arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Second(1))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + Dates.Second(100))) ==
            Dates.DateTime(1999, 12, 27, 0, 1, 40)
        @test Dates.DateTime(@jit(dt + Dates.Second(1000))) ==
            Dates.DateTime(1999, 12, 27, 0, 16, 40)
        @test Dates.DateTime(@jit(dt - Dates.Second(1))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59)
        @test Dates.DateTime(@jit(dt - Dates.Second(100))) ==
            Dates.DateTime(1999, 12, 26, 23, 58, 20)
        @test Dates.DateTime(@jit(dt - Dates.Second(1000))) ==
            Dates.DateTime(1999, 12, 26, 23, 43, 20)
    end

    @testset "DateTime-Millisecond arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Millisecond(1))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + Dates.Millisecond(100))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 100)
        @test Dates.DateTime(@jit(dt + Dates.Millisecond(1000))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 1)
        @test Dates.DateTime(@jit(dt - Dates.Millisecond(1))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
        @test Dates.DateTime(@jit(dt - Dates.Millisecond(100))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 900)
        @test Dates.DateTime(@jit(dt - Dates.Millisecond(1000))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59)

        dt = Dates.DateTime(1972, 6, 30, 23, 59, 59)
        @test Dates.DateTime(@jit(dt + Dates.Millisecond(1))) ==
            Dates.DateTime(1972, 6, 30, 23, 59, 59, 1)
        @test Dates.DateTime(@jit(dt - Dates.Millisecond(1))) ==
            Dates.DateTime(1972, 6, 30, 23, 59, 58, 999)
        @test Dates.DateTime(@jit(dt + Dates.Millisecond(-1))) ==
            Dates.DateTime(1972, 6, 30, 23, 59, 58, 999)
    end

    @testset "DateTime-Microsecond arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Microsecond(1))) == dt
        @test Dates.DateTime(@jit(dt + Dates.Microsecond(501))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + Dates.Microsecond(1499))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt - Dates.Microsecond(1))) == dt
        @test Dates.DateTime(@jit(dt - Dates.Microsecond(501))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
        @test Dates.DateTime(@jit(dt - Dates.Microsecond(1499))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
    end

    @testset "DateTime-Nanosecond arithmetic" begin
        dt = Dates.DateTime(1999, 12, 27)
        @test Dates.DateTime(@jit(dt + Dates.Nanosecond(1))) == dt
        @test Dates.DateTime(@jit(dt + Dates.Nanosecond(500_001))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt + Dates.Nanosecond(1_499_999))) ==
            Dates.DateTime(1999, 12, 27, 0, 0, 0, 1)
        @test Dates.DateTime(@jit(dt - Dates.Nanosecond(1))) == dt
        @test Dates.DateTime(@jit(dt - Dates.Nanosecond(500_001))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
        @test Dates.DateTime(@jit(dt - Dates.Nanosecond(1_499_999))) ==
            Dates.DateTime(1999, 12, 26, 23, 59, 59, 999)
    end

    @testset "Time-TimePeriod arithmetic" begin
        t = Dates.Time(0)
        @test convert(Dates.Time, @jit(t + Dates.Hour(1))) == Dates.Time(1)
        @test convert(Dates.Time, @jit(t - Dates.Hour(1))) == Dates.Time(23)
        @test convert(Dates.Time, @jit(t - Dates.Nanosecond(1))) ==
            Dates.Time(23, 59, 59, 999, 999, 999)
        @test convert(Dates.Time, @jit(t + Dates.Nanosecond(-1))) ==
            Dates.Time(23, 59, 59, 999, 999, 999)
        @test convert(Dates.Time, @jit(t + Dates.Hour(24))) == t
        @test convert(Dates.Time, @jit(t + Dates.Nanosecond(86400000000000))) == t
        @test convert(Dates.Time, @jit(t - Dates.Nanosecond(86400000000000))) == t
        @test convert(Dates.Time, @jit(t + Dates.Minute(1))) == Dates.Time(0, 1)
        @test convert(Dates.Time, @jit(t + Dates.Second(1))) == Dates.Time(0, 0, 1)
        @test convert(Dates.Time, @jit(t + Dates.Millisecond(1))) == Dates.Time(0, 0, 0, 1)
        @test convert(Dates.Time, @jit(t + Dates.Microsecond(1))) ==
            Dates.Time(0, 0, 0, 0, 1)
    end

    @testset "Commutativity" begin
        dt = Dates.DateTime(2000)
        d = Dates.Date(2000)
        t = Dates.Time(12)

        @test Dates.DateTime(@jit(Dates.Year(1) + dt)) ==
            Dates.DateTime(@jit(dt + Dates.Year(1)))
        @test Dates.DateTime(@jit(Dates.Month(1) + dt)) ==
            Dates.DateTime(@jit(dt + Dates.Month(1)))
        @test Dates.DateTime(@jit(Dates.Day(1) + dt)) ==
            Dates.DateTime(@jit(dt + Dates.Day(1)))
        @test Dates.DateTime(@jit(Dates.Hour(1) + dt)) ==
            Dates.DateTime(@jit(dt + Dates.Hour(1)))

        @test Dates.Date(@jit(Dates.Year(1) + d)) == Dates.Date(@jit(d + Dates.Year(1)))
        @test Dates.Date(@jit(Dates.Month(1) + d)) == Dates.Date(@jit(d + Dates.Month(1)))
        @test Dates.Date(@jit(Dates.Week(1) + d)) == Dates.Date(@jit(d + Dates.Week(1)))
        @test Dates.Date(@jit(Dates.Day(1) + d)) == Dates.Date(@jit(d + Dates.Day(1)))

        @test convert(Dates.Time, @jit(Dates.Hour(1) + t)) ==
            convert(Dates.Time, @jit(t + Dates.Hour(1)))
        @test convert(Dates.Time, @jit(Dates.Minute(1) + t)) ==
            convert(Dates.Time, @jit(t + Dates.Minute(1)))
        @test convert(Dates.Time, @jit(Dates.Second(1) + t)) ==
            convert(Dates.Time, @jit(t + Dates.Second(1)))
        @test convert(Dates.Time, @jit(Dates.Nanosecond(1) + t)) ==
            convert(Dates.Time, @jit(t + Dates.Nanosecond(1)))
    end

    @testset "Month arithmetic non-associativity" begin
        a = Dates.Date(2012, 1, 29)
        @test Dates.Date(@jit((a + Dates.Day(1)) + Dates.Month(1))) !=
            Dates.Date(@jit((a + Dates.Month(1)) + Dates.Day(1)))
        a = Dates.Date(2012, 1, 30)
        @test Dates.Date(@jit((a + Dates.Day(1)) + Dates.Month(1))) !=
            Dates.Date(@jit((a + Dates.Month(1)) + Dates.Day(1)))
        a = Dates.Date(2012, 2, 29)
        @test Dates.Date(@jit((a + Dates.Day(1)) + Dates.Month(1))) !=
            Dates.Date(@jit((a + Dates.Month(1)) + Dates.Day(1)))
    end

    @testset "toms and tons" begin
        # toms
        @test @jit(Dates.toms(Dates.Millisecond(1))) == 1
        @test @jit(Dates.toms(Dates.Second(1))) == 1000
        @test @jit(Dates.toms(Dates.Minute(1))) == 60000
        @test @jit(Dates.toms(Dates.Hour(1))) == 3600000
        @test @jit(Dates.toms(Dates.Day(1))) == 86400000
        @test @jit(Dates.toms(Dates.Week(1))) == 604800000

        # tons
        @test @jit(Dates.tons(Dates.Nanosecond(1))) == 1
        @test @jit(Dates.tons(Dates.Microsecond(1))) == 1000
        @test @jit(Dates.tons(Dates.Millisecond(1))) == 1000000
        @test @jit(Dates.tons(Dates.Second(1))) == 1000000000
        @test @jit(Dates.tons(Dates.Minute(1))) == 60000000000
        @test @jit(Dates.tons(Dates.Hour(1))) == 3600000000000

        # days
        @test @jit(Dates.days(Dates.Day(1))) == 1
        @test @jit(Dates.days(Dates.Week(1))) == 7
        @test @jit(Dates.days(Dates.Week(2))) == 14
    end

    @testset "datetime2julian" begin
        @test @jit(Dates.datetime2julian(Dates.DateTime(2000))) ==
            Dates.datetime2julian(Dates.DateTime(2000))
        @test @jit(Dates.datetime2julian(Dates.DateTime(1930, 12, 1, 1, 5, 1))) ==
            Dates.datetime2julian(Dates.DateTime(1930, 12, 1, 1, 5, 1))
    end

    @testset "Minimal timestepper with Dates" begin
        # Inspired by the usage in SpeedyWeather.jl

        # Minimal clock-like mutable struct
        mutable struct Clock{I,T,TS}
            n_timesteps::I
            time::T
            time_step::TS
        end

        # Minimal state
        struct State{T,C}
            x::T
            clock::C
        end

        function timestepping!(state::State)
            (; clock) = state

            @trace for i in 1:(clock.n_timesteps)
                timestep!(state)
            end

            return nothing
        end

        function timestep!(state::State)
            state.x .+= 1.0
            return state.clock.time += state.clock.time_step
        end

        clock = Clock(5, DateTime(2002, 1, 1), Dates.Day(1))
        state = State(zeros(1), clock)
        timestepping!(state)

        clock_jit = Clock(5, DateTime(2002, 1, 1), Dates.Day(1))
        state_jit = State(zeros(1), clock)
        @jit timestepping!(state_jit)

        @test DateTime(state_jit.clock.time) == state.clock.time
    end
end # @testset "ReactantDatesExt"
