from messages_dataframe import load_dataframes
import altair as alt
import polars as pl
import polars.selectors as cs
from pathlib import Path

alt.renderers.enable("browser")
alt.data_transformers.disable_max_rows()


def messages_over_time(messages: pl.DataFrame) -> pl.DataFrame:
    """Count messages in calendar-month buckets for a temporal histogram."""
    return (
        messages.lazy()
        .filter(pl.col("timestamp").is_not_null())
        .filter(
            pl.col("channel").is_in(qter_frames.channels["channel_id"]).not_()
        )
        .with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
        .group_by("month", "author_id")
        .agg(pl.len().alias("message_count"))
        .join(frames.users.lazy(), left_on="author_id", right_on="user_id")
        .with_columns(
            pl.col("nicknames").list.first().alias("username"),
            pl.col("message_count")
            .sum()
            .over("author_id")
            .alias("total_message_count"),
            pl.col("message_count")
            .diff()
            .over("author_id", order_by="month")
            .alias("message_count_difference"),
        )
        .sort("month", "total_message_count", descending=[False, True])
        .collect()
    )


frames = load_dataframes(Path("/home/henry/Documents/everything"))
qter_frames = load_dataframes()
# frames = load_dataframes()
#
# print(frames.messages.head())
# print(frames.channels.head())
# print(frames.users.head())


def plot_monthly_messages():
    monthly_messages = messages_over_time(frames.messages)

    (
        alt.Chart(monthly_messages)
        .mark_area()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y(
                "message_count:Q",
                title="Messages",
            ),
            color=alt.Color(
                "username:N",
                title="User",
                sort=alt.SortField(
                    field="total_message_count", order="descending"
                ),
            ),
            order=alt.Order("total_message_count:Q", sort="descending"),
            tooltip=[
                alt.Tooltip("month:T", title="Month"),
                "username:N",
                "message_count:Q",
            ],
        )
        .properties(title="Discord messages over time (Without Qter)")
        .show()
    )


def plot_change_in_monthly_messages():
    monthly_messages = (
        messages_over_time(frames.messages)
        .group_by("month")
        .agg(
            pl.all(),
            pos=pl.col("message_count_difference")
            .filter(pl.col("message_count_difference") > 0)
            .sum(),
            neg=pl.col("message_count_difference")
            .filter(pl.col("message_count_difference") < 0)
            .sum(),
        )
        .explode("username", "message_count_difference")
    )

    print(monthly_messages.head(50))

    (
        alt.Chart(monthly_messages)
        .mark_bar()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y(
                "neg:Q",
                title="Messages",
            ),
            y2="pos",
            color=alt.Color(
                "username:N",
                title="User",
                # sort=alt.SortField(field="message_count_difference", order="descending"),
            ),
            tooltip=[
                alt.Tooltip("month:T", title="Month"),
                "username:N",
                "message_count_difference:Q",
            ],
            order=alt.Order("total_message_count:Q", sort="descending"),
        )
        .properties(
            title="Change in Discord messages over time (Without Qter)"
        )
        .show()
    )


def plot_users_pie():
    data = (
        frames.messages.lazy()
        .filter(pl.col("author_id").is_not_null())
        .group_by("author_id")
        .agg(pl.len().alias("message_count"))
        .join(frames.users.lazy(), left_on="author_id", right_on="user_id")
        .with_columns(pl.col("nicknames").list.first().alias("username"))
        .sort(by=pl.col("message_count"), descending=True)
        .collect()
    )

    print(data.head())

    (
        alt.Chart(data)
        .mark_arc()
        .encode(
            theta=alt.Theta("message_count:Q"),
            color=alt.Color(
                "username:N",
                title="User",
                sort=alt.SortField(field="message_count", order="descending"),
            ),
            order=alt.Order("message_count:Q", sort="descending"),
            tooltip=["username:N", "message_count:Q"],
        )
        .properties(title="Messages by user")
        .show()
    )


# plot_users_pie()
plot_change_in_monthly_messages()
